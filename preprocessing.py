"""
preprocessing.py
----------------
Moduł realizujący potok inżynierii cech (Feature Engineering) dedykowany dla modeli 
szeregów czasowych DL (LSTM/Transformers). Spełnia interfejs standardowych Transformerów scikit-learn (narzucając z góry metody `.fit()` i `.transform()`).
Z uwagi na wagę gigabajtów operacyjnych, wewnętrznie wykorzystuje bezbłędny, wielowątkowy i mało-pamięciożerny silnik `Polars` - wejścia mogą być przesyłane leniwie (jako plik LazyFrame) redukując zużycie do zera.
"""

import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, weather_path='data/weather.csv', lags=[1, 2, 3, 6, 12, 24], global_stats_path=None):
        """
        Klasa inicjalizująca przepływ z parametrami wejściowymi.
        """
        self.weather_path = weather_path
        self.lags = lags
        self.global_stats_path = global_stats_path
        self.eps = 1e-6
        # Cache wyliczanych parametrów
        self.means_ = None
        self.stds_ = None
        self.numeric_cols_ = None

    def _build_features(self, X):
        """
        Rdzeń transformujący. Aplikuje asof joins dla pogody, opóźnienia, 
        różnicowanie pierwszego stopnia oraz ujęcia wariancji i anomalii.
        """
        # Gwarancja leniwego API do obsługi masywnych plików bez przeciążania RAM
        X_lz = X.lazy() if isinstance(X, pl.DataFrame) else X
        
        # 1. Konwersja czasu z ucinaniem stref (+00:00) ew. zastąpieniem znaku "T" przy stringach z Pandas
        X_lz = X_lz.with_columns(
            pl.col('timedate').str.replace("T", " ").str.slice(0, 19).str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False).alias('dt_obj')
        ).sort(['deviceId', 'dt_obj'])
        
        # 2. Połączenie pogodowe z API (Asof Join Backward) z dokładnością do 5min w dół
        if self.weather_path is not None:
            weather_lz = pl.scan_csv(self.weather_path)
            
            # Formatyzacja daty pobranej pogody (ucinamy strefy)
            weather_lz = weather_lz.with_columns(
                pl.col('timedate').str.replace("T", " ").str.slice(0, 19).str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False).alias('dt_obj')
            ).sort(['deviceId', 'dt_obj'])
            
            if 'timedate' in weather_lz.columns:
                weather_lz = weather_lz.drop('timedate')
                
            X_lz = X_lz.join_asof(
                weather_lz, 
                on='dt_obj', 
                by='deviceId', 
                strategy='backward'
            )

        # 3. Cecha czasu cyklicznego na bazie Datetime
        X_lz = X_lz.with_columns([
            pl.col('dt_obj').dt.hour().alias('hour'),
            pl.col('dt_obj').dt.weekday().alias('day_of_week')
        ]).with_columns([
            (np.sin(2 * np.pi * pl.col('hour') / 24)).alias('hour_sin'),
            (np.cos(2 * np.pi * pl.col('hour') / 24)).alias('hour_cos'),
            (np.sin(2 * np.pi * pl.col('day_of_week') / 7)).alias('day_sin'),
            (np.cos(2 * np.pi * pl.col('day_of_week') / 7)).alias('day_cos'),
            (pl.col('day_of_week') >= 6).cast(pl.Int8).alias('is_weekend')
        ])

        # 4. Różnicowanie ARIMA (Stacjonarność) i Logarytmiczne zwroty giełdowe chroniące skale w czasie
        t_cols = ['t1']
        diff_exprs = [pl.col(c).diff().over('deviceId').alias(f'{c}_diff') for c in t_cols]
        log_ret_exprs = [
            ((pl.col(c) + self.eps).log() - (pl.col(c).shift(1) + self.eps).log()).over('deviceId').alias(f'{c}_log_return') 
            for c in ['x1', 'x3']
        ]
        X_lz = X_lz.with_columns(diff_exprs + log_ret_exprs)

        # 5. Okna Autoregresji (Lags na bazie konkluzji ACF/PACF)
        lag_exprs = [
            pl.col(c).shift(l).over('deviceId').alias(f'{c}_lag_{l}') 
            for c in ['x1', 'x3'] for l in self.lags
        ]
        X_lz = X_lz.with_columns(lag_exprs)

        # 6. Modelowanie Zmienności GARCH i Wygładzanie Spektralne (Bezpieczne Okna Wierszowe aby uniknąć InvalidOperationError)
        X_lz = X_lz.with_columns([
            pl.col('x1_log_return').rolling_std(window_size=12).over('deviceId').alias('volatility_x1_1h'),
            pl.col('x1_log_return').rolling_std(window_size=288).over('deviceId').alias('volatility_x1_24h'),
            pl.col('x1').rolling_mean(window_size=6).over('deviceId').alias('x1_smoothed_30m')
        ])

        # 7. Makro-sieciowe Anomalie (Globalizowane z pliku zewnetrznego aby ominąć problem chunkowania!)
        if self.global_stats_path is not None:
            # Ladujemy plik statystyczny uprzednio zmapowany na pełnym zbiorze
            global_stats = pl.scan_parquet(self.global_stats_path)
            X_lz = X_lz.join(global_stats, on='dt_obj', how='left')
            
        X_lz = X_lz.with_columns([
            (pl.col('x1_log_return').abs() > (3 * pl.col('volatility_x1_24h').fill_null(float('inf')))).cast(pl.Int8).alias('x1_anomaly_flag')
        ])

        # Oczyszczanie niepotrzebnych zmiennych metadanych ulatniajacych model
        X_lz = X_lz.drop(['hour', 'day_of_week'])
        
        return X_lz

    def fit(self, X, y=None):
        """
        Buduje wewnętrzny graf operacji leniwych i matrycuje (collect) tylko parametry "mean" i "std" do wykorzystania
        przez mechanizm `transform()` dla poprawnej bezwyciekowej normalizacji zbiorów train->test.
        X operuje na typach Polars (np. pl.scan_csv('...')).
        """
        print("Trwa budowanie matrycy wag i kalkulacja metryk normalizacyjnych głębokich wejść Z-Score. Może to potrwać minuty przy petabajtowych objętościach...")
        
        X_feats = self._build_features(X)
        
        # Ekstrakcja schematu zmiennych dla zabezpieczeń między wersjami 
        schema = X_feats.collect_schema() if hasattr(X_feats, 'collect_schema') else X_feats.schema
        
        # Omijanie zmiennych, których nie można/nie trzeba skalować (Target, flagi, stringi)
        exclude = ['deviceId', 'timedate', 'dt_obj', 'x2', 'is_weekend', 'x1_anomaly_flag']
        
        NUMERIC_DTYPES = (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        )
        
        self.numeric_cols_ = [
            col for col, dtype in schema.items() 
            if col not in exclude and dtype in NUMERIC_DTYPES
        ]
        
        stats_exprs = []
        for col in self.numeric_cols_:
            stats_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            stats_exprs.append(pl.col(col).std().alias(f"{col}_std"))
        
        # Ulepszona optymalizacja RAMowa: Odpal collect() na ramce wykonującej w locie wylistowane średnie, ignorując całą zawartość 5 GB x 120 Featursów
        stats_df = X_feats.select(stats_exprs).collect()
        
        self.means_ = {col: stats_df.get_column(f"{col}_mean")[0] for col in self.numeric_cols_}
        self.stds_ =  {col: stats_df.get_column(f"{col}_std")[0] for col in self.numeric_cols_}
        
        print(f"Baza Normalizacyjna ukończona! Posiadam zserializowane parametry skalujące dla {len(self.numeric_cols_)} cech numerycznych.")
        return self

    def transform(self, X, y=None):
        """
        Protokół wylistowujący zdefiniowane transformacje na zbiorze z gwarancją standaryzacji do Tensorów (Z=(X-mean)/std).
        Zwraca ramkę typu pl.LazyFrame. Należy ręcznie zwieńczyć pobranie funkcją `.collect()`.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Skalery są puste. Skrypt nie został zmapowany użyj `fit()` na zbiorze Trainingowym zanim wejdziesz w Inferencję `transform()` na Testowym!")
            
        X_feats = self._build_features(X)
        
        norm_exprs = []
        for col in self.numeric_cols_:
             std_val = self.stds_[col] if (self.stds_[col] is not None and self.stds_[col] != 0) else 1.0
             mean_val = self.means_[col] if self.means_[col] is not None else 0.0
             
             norm_exprs.append(
                 ((pl.col(col) - mean_val) / std_val).alias(col)
             )
        
        # Zastępuje naturalne stężenia znormalizowanym, wystandaryzowanym wagowo odpowiednikiem w tej samej kolumnie
        X_norm = X_feats.with_columns(norm_exprs)
        
        # Wypełnianie braków z przesunięć autoregresyjnych na zerowy wpływ na wagi w PyTorch (po Z-score, 0 to poprostu statystycznie najczęstsze oczekiwanie - wygasza gradient).
        X_norm = X_norm.fill_nan(0.0).fill_null(0.0)
        
        return X_norm


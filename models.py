import torch
import torch.nn as nn
import math

def get_device():
    """
    Funkcja pomocnicza. Gwarantuje przerzucenie wektorów na układ chłodzenia CUDA jeśli tylko jest na tej maszynie dostępny.
    Wspiera ewentualne macki z MPS dla procesorów ARM, a jeśli nie - spada do CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # Dodatkowe wsparcie pod procesory graficzne Apple-Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")

# =========================================================================
# ARCHITEKTURA 1: Long Short-Term Memory (LSTM)
# =========================================================================

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Klasyczna Sieć Rekurencyjna z ukrytą w sobie komórką pamięci.
        Argument `batch_first=True` oznacza, że sieć akceptuje Tensory w uniwersalnym 
        kształcie wejścia (batch_size, sequence_length, features).
        """
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Główny blok pamięci długo/krótkoterminowej
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Ostatnia warstwa zrzucająca n-wymiarową wiedzę do wymaganego rozmiaru odpowiedzi (output_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Tworzenie tensorów stanów ukrytych.
        # Niezwykle ważne: upewniamy się poprzez `.to(x.device)` ze wektory te lądują na karcie graficznej (CUDA), a nie w RAMie
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Główna pętla uczenia na LSTM
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Wyciągamy ostatni krok czasowy sekwencji ("podsumowanie" całego upływającego czasu dla urządzenia)
        out = out[:, -1, :]
        
        # Warunek finalny - Dense Layer
        predictions = self.fc(out)
        
        return predictions


# =========================================================================
# ARCHITEKTURA 2: Transformer (Encoder-based)
# =========================================================================

class PositionalEncoding(nn.Module):
    """
    Transformery nie analizują czasu, nie mają pojęcia o kolejności zdarzeń w szeregu.
    Pełnią rolę wtryskiwaczy wektorów gęstościowych fal sine/cosine, by poinformować model,
    który token pochodzi z którego punktu czasowego.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Uformowanie do zderzenia z wtykami `batch_first`
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Dodanie sygnału częstotliwości pozycyjnej bezpośrednio do cech wejściowych
        x = x + self.pe[:, :x.size(1), :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=256, dropout=0.1, max_seq_len=1000):
        """
        Zoptymalizowany pod GPU Transformer. Idealny do dalekich i chaotycznych interwencji, 
        gdzie potrafi w jednej chwili używając mechanizmów `Attention` przenieść wagę 
        do konkretnego piku sprzed kilkunastu godzin wstecz ignorując bezwzględny szum 
        ostatnich minut na stacji pomiarowej.
        """
        super(TimeSeriesTransformer, self).__init__()
        
        # Zmienia liczbę Twoich customowych featerów (np 50 numerycznych kolumn) do stałej 
        # wielkości `d_model` oczekiwanej na wejście przez blok Self-Attention (np. 128)
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Moduł Centralny (Koder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True  # Przetwarzanie Cuda zdominowane wektorem równoległym
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sieć dekodująca do zwrócenia jednej lub prededifniowanej zmiennej docelowej (np. x2 mean)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_size)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) -> na CUDA
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        
        # Przetłoczenie Attention
        # out shape: (batch_size, seq_len, d_model)
        out = self.transformer_encoder(x)
        
        # Pule sekwencji. Tradycyjnie przy Transformersach z Time Series uśrednia sie Globalnym Poolingiem. 
        # Redukuje to sekwencję [seq_len] przez zsumowanie uwagi w dół po wymiarze 1.
        out_pooled = out.mean(dim=1)
        
        predictions = self.fc(out_pooled)
        return predictions

if __name__ == "__main__":
    device = get_device()
    print(f"Silnik graficzny do Deep Learningu ustabilizowany na: {device}")
    
    # Podglądowy proof of concept dla LSTM
    model_lstm = TimeSeriesLSTM(input_size=40, hidden_size=64, num_layers=2, output_size=1)
    model_lstm.to(device) # Rzutowanie parametrów modelu na procesor CUDA
    print(model_lstm)
    
    # Podglądowy proof of concept dla Transformera
    model_transformer = TimeSeriesTransformer(input_size=40, d_model=128, nhead=4, num_layers=3, output_size=1)
    model_transformer.to(device) 
    print("\\nModele osadzone w środowisku!")

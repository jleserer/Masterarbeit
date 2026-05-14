"""
Informer Model for Stock Price Prediction
==========================================
Architecture based on Zhou et al. (2021) - "Informer: Beyond Efficient
Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper).

Key innovations:
  - ProbSparse Self-Attention: O(L log L) complexity instead of O(L^2)
  - Self-Attention Distilling: Halves cascading layer input via Conv+MaxPool
  - Generative Decoder: Single forward pass for output generation

Adapted for single-step prediction (pred_len=1) of log returns.

Lookback Window: L=60 (60 trading days history)
Target: Single-output (Close log return)
Framework: PyTorch
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data_preparation import DataPreparator, create_sequences


# =============================================================================
# INFORMER ARCHITECTURE COMPONENTS
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DataEmbedding(nn.Module):
    """Projects input features to d_model dimensions with positional encoding."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_encoding(x)
        return self.dropout(x)


class ProbAttention(nn.Module):
    """ProbSparse Self-Attention Mechanism (Zhou et al., 2021).

    Selects top-u queries based on KL-divergence measurement,
    reducing complexity from O(L^2) to O(L log L).
    """

    def __init__(self, mask_flag=False, factor=5, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """Calculate the sparse attention score."""
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate sampled Q_K: (B, H, L_Q, sample_k)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)

        # Random sampling of keys
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Q_K_sample: (B, H, L_Q, sample_k)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find top-u queries based on sparsity measurement
        # M = max(Q_K) - mean(Q_K) → measures "sharpness" of attention
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # Top-u queries with highest M scores
        M_top = M.topk(n_top, sorted=False)[1]

        return M_top

    def forward(self, queries, keys, values, attn_mask=None):
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        U_part = max(1, self.factor * int(math.ceil(math.log(L_K + 1))))
        u = max(1, self.factor * int(math.ceil(math.log(L_Q + 1))))

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # ProbSparse: select top-u queries
        scores_top_index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Build sparse attention scores
        # Initialize with mean value (for non-selected queries)
        scale = 1.0 / math.sqrt(D)
        context = values.mean(dim=-2, keepdim=True).expand(B, H, L_Q, values.size(-1)).clone()

        # Calculate attention for top-u queries
        Q_selected = queries[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            scores_top_index
        ]  # (B, H, u, D)

        attn_scores = torch.matmul(Q_selected, keys.transpose(-2, -1)) * scale  # (B, H, u, L_K)

        if self.mask_flag and attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, float('-inf'))

        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        attn_output = torch.matmul(attn_probs, values)  # (B, H, u, D)

        # Scatter back into full context
        context[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            scores_top_index
        ] = attn_output

        return context


class AttentionLayer(nn.Module):
    """Multi-Head Attention wrapper with projection layers."""

    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L_Q, H, -1).transpose(1, 2)
        keys = self.key_projection(keys).view(B, L_K, H, -1).transpose(1, 2)
        values = self.value_projection(values).view(B, L_K, H, -1).transpose(1, 2)

        out = self.inner_attention(queries, keys, values, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, L_Q, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    """Transformer encoder layer: Attention + Feed-Forward + LayerNorm."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        y = x = self.norm1(x)

        # Feed-forward with residual (using Conv1d for efficiency)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class ConvLayer(nn.Module):
    """Distilling layer: Conv1D + BatchNorm + ELU + MaxPool.

    Halves the sequence length between encoder layers (attention distilling).
    """

    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in, out_channels=c_in,
            kernel_size=3, padding=1, padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class Encoder(nn.Module):
    """Informer encoder with attention distilling."""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer: Self-Attention + Cross-Attention + Feed-Forward."""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)

        # Cross-attention with encoder output
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))
        y = x = self.norm2(x)

        # Feed-forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    """Informer decoder with stacked decoder layers."""

    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Informer(nn.Module):
    """Full Informer model for time series forecasting.

    Args:
        enc_in: Number of input features (encoder)
        dec_in: Number of input features (decoder)
        c_out: Number of output features
        seq_len: Input sequence length (lookback window L)
        label_len: Decoder start token length (L // 2)
        pred_len: Prediction horizon (1 for single-step)
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        factor: ProbSparse attention factor
        activation: Activation function ('gelu' or 'relu')
        distil: Whether to use attention distilling
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model=64, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=256, dropout=0.05, factor=5, activation='gelu', distil=True):
        """seq_len wird aus Backward-Compat weiter akzeptiert, aber nicht
        gespeichert oder benutzt. Die tatsächliche Sequenzlänge wird zur
        Laufzeit aus der Tensor-Shape in ProbAttention.forward gelesen
        (B, H, L_Q, D = queries.shape), d.h. der Lookback-Sweep wirkt
        korrekt pro L ohne dass das Modell einen festen seq_len bräuchte.
        """
        super().__init__()
        # seq_len bewusst NICHT gespeichert: unbenutzt und irreführend.
        self.label_len = label_len
        self.pred_len = pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(mask_flag=False, factor=factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            conv_layers=[ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(mask_flag=True, factor=factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    AttentionLayer(
                        ProbAttention(mask_flag=False, factor=factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Output projection
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc: (batch, seq_len, enc_in) - encoder input
            x_dec: (batch, label_len + pred_len, dec_in) - decoder input
        Returns:
            (batch, pred_len, c_out) - predictions
        """
        enc_out = self.encoder(self.enc_embedding(x_enc))
        dec_out = self.decoder(self.dec_embedding(x_dec), enc_out)
        out = self.projection(dec_out)
        return out[:, -self.pred_len:, :]  # Only return prediction part


# =============================================================================
# INFORMER MODEL WRAPPER (same interface as LSTMModel / CNNModel / GRUModel)
# =============================================================================

class InformerModel:
    """Informer-based stock price predictor (PyTorch)."""

    # Architecture Configuration
    LOOKBACK_WINDOW = 60
    D_MODEL = 64
    N_HEADS = 8
    E_LAYERS = 2
    D_LAYERS = 1
    D_FF = 256  # 4 * D_MODEL
    DROPOUT = 0.05
    FACTOR = 5
    ACTIVATION = 'gelu'
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 100

    def __init__(self, data_path, output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parameter_tuning', 'results', 'INFORMER', 'SP500')):
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.preparator = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.predictions = {}
        self.device = torch.device('cpu')

    def prepare_data(self):
        """Load and prepare data using DataPreparator."""
        self.preparator = DataPreparator(self.data_path)
        train_data, val_data, test_data = self.preparator.load_and_prepare()

        self.X_train, self.y_train = create_sequences(train_data, self.LOOKBACK_WINDOW)
        self.X_val, self.y_val = create_sequences(val_data, self.LOOKBACK_WINDOW)
        self.X_test, self.y_test = create_sequences(test_data, self.LOOKBACK_WINDOW)

        print(f"\nSequence Shapes (Lookback Window L={self.LOOKBACK_WINDOW}):")
        print(f"  X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        print(f"  X_val:   {self.X_val.shape}, y_val:   {self.y_val.shape}")
        print(f"  X_test:  {self.X_test.shape}, y_test:  {self.y_test.shape}")

    def build_model(self):
        """Build Informer model."""
        n_features = self.X_train.shape[2]
        label_len = max(1, self.LOOKBACK_WINDOW // 2)  # P1-8: min. 1 reales Decoder-Token

        self.model = Informer(
            enc_in=n_features,
            dec_in=n_features,
            c_out=1,
            seq_len=self.LOOKBACK_WINDOW,
            label_len=label_len,
            pred_len=1,
            d_model=self.D_MODEL,
            n_heads=self.N_HEADS,
            e_layers=self.E_LAYERS,
            d_layers=self.D_LAYERS,
            d_ff=self.D_FF,
            dropout=self.DROPOUT,
            factor=self.FACTOR,
            activation=self.ACTIVATION,
            distil=True
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nInformer Model Built:")
        print(f"  d_model={self.D_MODEL}, n_heads={self.N_HEADS}, "
              f"e_layers={self.E_LAYERS}, d_layers={self.D_LAYERS}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def _prepare_decoder_input(self, X_batch):
        """Construct decoder input for single-step prediction.

        Decoder input = last label_len timesteps from encoder input + zero padding for pred_len.
        """
        label_len = max(1, self.LOOKBACK_WINDOW // 2)  # P1-8: min. 1 reales Decoder-Token
        # Take the last label_len timesteps as decoder start token
        dec_start = X_batch[:, -label_len:, :]  # (batch, label_len, n_features)
        # Append zero padding for prediction position
        pred_padding = torch.zeros(X_batch.size(0), 1, X_batch.size(2), device=self.device)
        dec_input = torch.cat([dec_start, pred_padding], dim=1)  # (batch, label_len+1, n_features)
        return dec_input

    def train(self):
        """Train the Informer model with PyTorch training loop."""
        print(f"\nTraining Informer ({self.EPOCHS} epochs, batch_size={self.BATCH_SIZE})...")

        # Convert to tensors
        X_train_t = torch.FloatTensor(self.X_train).to(self.device)
        y_train_t = torch.FloatTensor(self.y_train).to(self.device)
        X_val_t = torch.FloatTensor(self.X_val).to(self.device)
        y_val_t = torch.FloatTensor(self.y_val).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.EPOCHS):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                dec_input = self._prepare_decoder_input(X_batch)
                output = self.model(X_batch, dec_input)  # (batch, 1, 1)
                pred = output.squeeze(-1).squeeze(-1)  # (batch,)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                dec_input_val = self._prepare_decoder_input(X_val_t)
                val_output = self.model(X_val_t, dec_input_val)
                val_pred = val_output.squeeze(-1).squeeze(-1)
                val_loss = criterion(val_pred, y_val_t).item()
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.EPOCHS}: "
                      f"Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")

    def evaluate(self):
        """Evaluate model on all splits. Returns (test_loss, test_mae, test_rmse)."""
        self.model.eval()
        criterion = nn.MSELoss()

        results = {}
        for split_name, X_data, y_data in [
            ('train', self.X_train, self.y_train),
            ('val', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test),
        ]:
            X_t = torch.FloatTensor(X_data).to(self.device)
            y_t = torch.FloatTensor(y_data).to(self.device)

            with torch.no_grad():
                dec_input = self._prepare_decoder_input(X_t)
                output = self.model(X_t, dec_input)
                pred = output.squeeze(-1).squeeze(-1)
                loss = criterion(pred, y_t).item()
                mae = torch.mean(torch.abs(pred - y_t)).item()
                rmse = np.sqrt(loss)

            results[split_name] = (loss, mae, rmse)

        train_loss, train_mae, train_rmse = results['train']
        val_loss, val_mae, val_rmse = results['val']
        test_loss, test_mae, test_rmse = results['test']

        print(f"\nTraining   Loss: {train_loss:.6f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        print(f"Test       Loss: {test_loss:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        return test_loss, test_mae, test_rmse

    def predict(self):
        """Generate predictions and inverse-transform to original prices."""
        self.model.eval()
        self.predictions = {}

        for split_name, X_data, y_data in [
            ('train', self.X_train, self.y_train),
            ('val', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test),
        ]:
            X_t = torch.FloatTensor(X_data).to(self.device)
            with torch.no_grad():
                dec_input = self._prepare_decoder_input(X_t)
                output = self.model(X_t, dec_input)
                pred = output.squeeze(-1).squeeze(-1).numpy()

            pred_prices = self.preparator.inverse_transform(
                pred.reshape(-1, 1), split=split_name, lookback=self.LOOKBACK_WINDOW
            )
            actual_prices = self.preparator.inverse_transform(
                y_data.reshape(-1, 1), split=split_name, lookback=self.LOOKBACK_WINDOW
            )

            self.predictions[split_name] = {
                'predicted_returns': pred,
                'actual_returns': y_data,
                'predicted_prices': pred_prices,
                'actual_prices': actual_prices,
            }

        print(f"\nPrediction Price Ranges:")
        for split_name in ['train', 'val', 'test']:
            p = self.predictions[split_name]
            print(f"  {split_name}: Actual [{p['actual_prices'].min():.2f}, {p['actual_prices'].max():.2f}]"
                  f"  Predicted [{p['predicted_prices'].min():.2f}, {p['predicted_prices'].max():.2f}]")

    def plot_results(self):
        """Plot and save training history."""
        os.makedirs(self.output_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.history['train_loss'], label='Training Loss')
        ax.plot(self.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Informer Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=150)
        plt.close()

    def save_model(self):
        """Save model state dict."""
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = os.path.join(self.output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def run_full_pipeline(self):
        """Execute full training pipeline."""
        print("=" * 70)
        print("INFORMER MODEL - Full Pipeline")
        print("=" * 70)

        self.prepare_data()
        self.build_model()
        self.train()
        self.evaluate()
        self.predict()
        self.plot_results()
        self.save_model()

        print("\nPipeline completed!")


# =============================================================================
# HELPER FUNCTIONS (for use by parameter_tuning.py and sweep)
# =============================================================================

def build_informer(lookback_window, n_features, config):
    """Build an Informer model from a config dict.

    Args:
        lookback_window: Sequence length L
        n_features: Number of input features
        config: Dict with keys: d_model, n_heads, dropout, lr, batch, epochs
                Optional: e_layers, d_layers, factor

    Returns:
        Informer model instance (PyTorch nn.Module)
    """
    d_model = config.get('d_model', 64)
    n_heads = config.get('n_heads', 8)
    dropout = config.get('dropout', 0.05)
    e_layers = config.get('e_layers', 2)
    d_layers = config.get('d_layers', 1)
    d_ff = config.get('d_ff', 4 * d_model)
    factor = config.get('factor', 5)

    label_len = max(1, lookback_window // 2)  # P1-8: min. 1 reales Decoder-Token

    model = Informer(
        enc_in=n_features,
        dec_in=n_features,
        c_out=1,
        seq_len=lookback_window,
        label_len=label_len,
        pred_len=1,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        factor=factor,
        activation='gelu',
        distil=True
    )
    return model


def train_informer(model, X_train, y_train, config, lookback_window, verbose=0,
                   X_val=None, y_val=None):
    """Train an Informer model. Returns training history dict.

    Semantik der Val-Metriken:
      - Wenn X_val/y_val übergeben: val_loss/val_mae werden nach jeder Epoche
        berechnet (analog zu Keras model.fit(..., validation_data=...)).
      - Ohne X_val/y_val: history enthält nur train_loss pro Epoche;
        Val-Metriken müssen dann separat via evaluate_informer() berechnet werden.

    Args:
        model: Informer nn.Module
        X_train: numpy array (n, L, features)
        y_train: numpy array (n,)
        config: Dict with 'lr', 'batch', 'epochs'
        lookback_window: L (for decoder input construction)
        verbose: 0=silent, 1=epoch logs
        X_val, y_val: Optional Val-Set für per-Epoche Val-Loss

    Returns:
        dict with 'train_loss' (+ optional 'val_loss', 'val_mae', 'mae') pro Epoche
    """
    device = next(model.parameters()).device
    label_len = max(1, lookback_window // 2)  # P1-8: min. 1 reales Decoder-Token

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config['batch'], shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    has_val = X_val is not None and y_val is not None
    # MARE wird nicht pro Epoche getrackt — sie ist auf der Preis-Ebene definiert
    # und bräuchte dafür Zugriff auf DataPreparator/base_prices, was hier nicht
    # sauber verfügbar ist. MARE wird post-training via evaluate_informer_model
    # aus dem Tuner auf den inverse-transformierten Preisen berechnet.
    history = {'train_loss': []}
    if has_val:
        history['val_loss'] = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            dec_start = X_batch[:, -label_len:, :]
            pred_padding = torch.zeros(X_batch.size(0), 1, X_batch.size(2), device=device)
            dec_input = torch.cat([dec_start, pred_padding], dim=1)

            output = model(X_batch, dec_input)
            pred = output.squeeze(-1).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(avg_loss)

        if has_val:
            val_loss, _val_mae = evaluate_informer(model, X_val, y_val, lookback_window)
            history['val_loss'].append(float(val_loss))

        if verbose and ((epoch + 1) % 10 == 0):
            msg = f"  Epoch {epoch+1}/{config['epochs']}: Loss={avg_loss:.6f}"
            if has_val:
                msg += f", Val Loss={history['val_loss'][-1]:.6f}"
            print(msg)

    return history


MARE_EPSILON = 1e-6


def evaluate_informer(model, X_data, y_data, lookback_window, return_mare=False):
    """Evaluate an Informer model.

    Args:
        model: Informer nn.Module
        X_data: numpy array (n, L, features)
        y_data: numpy array (n,)
        lookback_window: L
        return_mare: Wenn True, wird zusätzlich MARE zurückgegeben.
                     Für Backward-Compat ist der Default False — bestehende
                     Aufrufer brauchen nicht angepasst zu werden.

    Returns:
        (mse_loss, mae) oder (mse_loss, mae, mare) wenn return_mare=True.
        MARE = mean(|pred - target| / (|target| + eps)) — analog
        torchmetrics.MeanAbsoluteRelativeError.
    """
    device = next(model.parameters()).device
    label_len = max(1, lookback_window // 2)  # P1-8: min. 1 reales Decoder-Token

    model.eval()
    X_t = torch.FloatTensor(X_data).to(device)
    y_t = torch.FloatTensor(y_data).to(device)

    with torch.no_grad():
        dec_start = X_t[:, -label_len:, :]
        pred_padding = torch.zeros(X_t.size(0), 1, X_t.size(2), device=device)
        dec_input = torch.cat([dec_start, pred_padding], dim=1)

        output = model(X_t, dec_input)
        pred = output.squeeze(-1).squeeze(-1)

        mse_loss = F.mse_loss(pred, y_t).item()
        abs_err = torch.abs(pred - y_t)
        mae = torch.mean(abs_err).item()
        if return_mare:
            mare = torch.mean(abs_err / (torch.abs(y_t) + MARE_EPSILON)).item()
            return mse_loss, mae, mare

    return mse_loss, mae


def predict_informer(model, X_data, lookback_window):
    """Generate predictions with an Informer model.

    Args:
        model: Informer nn.Module
        X_data: numpy array (n, L, features)
        lookback_window: L

    Returns:
        numpy array of predictions (n,)
    """
    device = next(model.parameters()).device
    label_len = max(1, lookback_window // 2)  # P1-8: min. 1 reales Decoder-Token

    model.eval()
    X_t = torch.FloatTensor(X_data).to(device)

    with torch.no_grad():
        dec_start = X_t[:, -label_len:, :]
        pred_padding = torch.zeros(X_t.size(0), 1, X_t.size(2), device=device)
        dec_input = torch.cat([dec_start, pred_padding], dim=1)

        output = model(X_t, dec_input)
        pred = output.squeeze(-1).squeeze(-1)

    return pred.cpu().numpy()


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData', 'preprocessedData', 'SP500_historical_data.csv')

    model = InformerModel(data_path)
    model.run_full_pipeline()

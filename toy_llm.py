# toy_llm.py --n_layers 6 --epochs 100
# 100% in epoch 44, 5s/epoch
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import random
import argparse
import math
import time


def generate_ab(ndigits):
    while True:
        a = random.randint(10**(ndigits-1), 10**ndigits - 1)
        b = random.randint(10**(ndigits-1), 10**ndigits - 1)
        if a + b >= 10**ndigits:
            continue
        return a, b

# --- Configuration & Arguments ---
def get_args():
    parser = argparse.ArgumentParser(description="Toy LLM for Addition (masked transformer only)")
    parser.add_argument('--ndigits', type=int, default=4, help='Number of digits for addends (e.g., 4 for 1234+5678)')
    parser.add_argument('--train_size', type=int, default=10000, help='Number of training examples')
    parser.add_argument('--test_size', type=int, default=500, help='Number of test examples')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for transformer feedforward')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--n_olayers', type=int, default=6, help='Number of output layers')
    return parser.parse_args()

# --- Data Generation ---
class AdditionDataset(Dataset):
    def __init__(self, size, ndigits, vocab, device):
        self.size = size
        self.ndigits = ndigits
        self.vocab = vocab
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.device = device
        self.data = self._generate_data()


    def _generate_data(self):
        rows = []
        seen = set()
        
        # Structure: "1234+5678=3579" (Fixed length)
        # Length = ndigits + 1 + ndigits + 1 + (ndigits) = 3 * ndigits + 2
        
        self.seq_len = self.ndigits * 3 + 2
        
        while len(rows) < self.size:
            a, b = generate_ab(self.ndigits)
            res = a + b
            eqn = f"{a}+{b}={res}"

            if eqn in seen:
                continue
            
            seen.add(eqn)
            
            # Convert to indices
            indices = [self.char_to_idx[c] for c in eqn]
            rows.append(torch.tensor(indices, dtype=torch.long))
            
        # Stack once and move the entire dataset to the target device
        return torch.stack(rows, dim=0).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input: Sequence
        # Target: Same sequence shifted by 1 (standard autoregressive training)
        seq = self.data[idx]
        x = seq[:-1] # Inputs
        y = seq[1:]  # Targets
        return x, y

# --- Models ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, Dim] or [Batch, SeqLen, Dim]
        # We assume Batch First for the main model, but handle dimension accordingly
        return x + self.pe[:x.size(1), :].unsqueeze(0)


def _get_activation_fn(name_or_fn):
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn == "relu":
        return F.relu
    if name_or_fn == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation {name_or_fn}")


class MyTransformerEncoderLayer(nn.Module):
    """
    Custom Transformer encoder layer mirroring nn.TransformerEncoderLayer with batch_first support.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation="relu",
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            return self._forward_pre_norm(src, src_mask, src_key_padding_mask)
        return self._forward_post_norm(src, src_mask, src_key_padding_mask)

    def _forward_post_norm(self, src, src_mask, src_key_padding_mask):
        attn_out = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False
        )[0]
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src

    def _forward_pre_norm(self, src, src_mask, src_key_padding_mask):
        src_norm = self.norm1(src)
        attn_out = self.self_attn(
            src_norm, src_norm, src_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False
        )[0]
        src = src + self.dropout1(attn_out)
        src_norm = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_out)
        return src


class MyTransformerEncoder(nn.Module):
    """
    Minimal Transformer encoder stack so we can customize/inspect layers directly.
    Mirrors nn.TransformerEncoder behavior we rely on.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int, norm: nn.LayerNorm | None = None):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(copy.deepcopy(encoder_layer) for _ in range(num_layers))
        self.gru_cells = nn.ModuleList(
            nn.GRUCell(self.layers[0].d_model, self.layers[0].d_model) for _ in range(len(self.layers))
        )
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for idx, mod in enumerate(self.layers):
            layer_out = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            # GRU mixes previous layer output (hidden) with current layer output (input).
            b, s, d = layer_out.shape
            output = self.gru_cells[idx](
                layer_out.reshape(-1, d),
                output.reshape(-1, d),
            ).view(b, s, d)
        if self.norm is not None:
            output = self.norm(output)
        return output


class ToyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_heads, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1),
            persistent=False,
        )
        encoder_layer = MyTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = MyTransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def prepare_forward(self, x):
        # x shape: [Batch, SeqLen]
        # Causal Mask: Upper triangular is -inf
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        return emb, mask

    def forward(self, x):
        emb, mask = self.prepare_forward(x)
        # Transformer expects [Batch, Seq, Dim] because we set batch_first=True
        out = self.transformer(emb, mask=mask)
        out = out + emb  # retain top-level skip from embeddings to logits
        logits = self.fc_out(out)
        return logits

# --- Training & Evaluation ---

def evaluate_model(model, data, seq_len, ndigits, batch_size):
    model.eval()
    correct_eq = 0
    correct_chars = 0
    total_chars = 0
    prompt_len = seq_len - ndigits

    x_all = data[:, :-1]
    y_all = data[:, 1:]
    total = data.size(0)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            x = x_all[start:end]
            y = y_all[start:end]

            # Prompt is everything up to and including '='
            prompt = x[:, :prompt_len]
            generated = prompt

            for _ in range(ndigits):
                out = model(generated)
                next_tok = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)

            expected_full = torch.cat([x, y[:, -1:]], dim=1)
            matches = generated == expected_full

            correct_chars += matches.sum().item()
            total_chars += expected_full.numel()
            correct_eq += matches.all(dim=1).sum().item()
    
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    total_acc = correct_eq / total if total > 0 else 0.0
    return char_acc, total_acc

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Vocab: 0-9, +, =
    vocab = "0123456789+="
    
    # Dataset: generate once and split into train/test
    total_size = args.train_size + args.test_size
    full_dataset = AdditionDataset(total_size, args.ndigits, vocab, device)
    seq_len = full_dataset.seq_len
    train_dataset, test_dataset = random_split(
        full_dataset,
        [args.train_size, args.test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_indices = torch.tensor(train_dataset.indices, device=device, dtype=torch.long)
    test_indices = torch.tensor(test_dataset.indices, device=device, dtype=torch.long)
    train_data = full_dataset.data.index_select(0, train_indices)
    test_data = full_dataset.data.index_select(0, test_indices)
    train_inputs = train_data[:, :-1]
    train_targets = train_data[:, 1:]
    test_inputs = test_data[:, :-1]
    test_targets = test_data[:, 1:]
    
    # Model
    model = ToyLLM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=full_dataset.seq_len,  # For Positional Encoding
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    params_2d = [p for p in params if p.ndim == 2]
    params_other = [p for p in params if p.ndim != 2]
    optimizers = []
    if params_2d:
        optimizers.append(optim.Muon(params_2d, lr=args.lr))
    if params_other:
        # Muon only supports 2D parameters; fall back to Adam for biases, LayerNorm, etc.
        optimizers.append(optim.Adam(params_other, lr=args.lr))
    
    # Training Loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        perm = torch.randperm(train_inputs.size(0), device=device)
        for start in range(0, train_inputs.size(0), args.batch_size):
            end = min(start + args.batch_size, train_inputs.size(0))
            idx = perm[start:end]
            x = train_inputs[idx]
            y = train_targets[idx]
            emb, mask = model.prepare_forward(x)

            for _ in range(0, args.n_olayers):
                for opt in optimizers:
                    opt.zero_grad()
                
                emb = model.transformer(emb, mask=mask)
                output = model.fc_out(emb)
                
                # Output: [Batch, SeqLen, Vocab]
                # Target: [Batch, SeqLen]
                # Flatten for Loss
                loss = criterion(output.reshape(-1, len(vocab)), y.reshape(-1))
                
                loss.backward()
                for opt in optimizers:
                    opt.step()
                emb = emb.detach()

            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Evaluation: test loss
        model.eval()
        total_test_loss = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for start in range(0, test_inputs.size(0), args.batch_size):
                end = min(start + args.batch_size, test_inputs.size(0))
                x = test_inputs[start:end]
                y = test_targets[start:end]
                emb, mask = model.prepare_forward(x)
                for _ in range(0, args.n_olayers):
                    emb = model.transformer(emb, mask=mask)
                out = model.fc_out(emb)
                test_loss = criterion(out.reshape(-1, len(vocab)), y.reshape(-1))
                total_test_loss += test_loss.item()
                num_test_batches += 1
        avg_test_loss = total_test_loss / max(num_test_batches, 1)

        # Evaluation: generation-based accuracies (train and test)
        train_char_acc, train_total_acc = evaluate_model(
            model, train_data, seq_len, args.ndigits, args.batch_size
        )
        test_char_acc, test_total_acc = evaluate_model(
            model, test_data, seq_len, args.ndigits, args.batch_size
        )

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{args.epochs}, "
            f"Train Time: {epoch_time:.2f}s, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_test_loss:.4f}, "
            f"Train Char Acc: {train_char_acc*100:.2f}%, "
            f"Train Total Acc: {train_total_acc*100:.2f}%, "
            f"Test Char Acc: {test_char_acc*100:.2f}%, "
            f"Test Total Acc: {test_total_acc*100:.2f}%"
        )
    
if __name__ == "__main__":
    args = get_args()
    train(args)

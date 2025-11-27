# As toy_llm gives super bad results, toy_neurosat super good results, I need to 
#   start moving important ideas to a library that I can reuse from both.
# toy_llm.py --n_layers 6 --epochs 100
# 100% in epoch 44, 5s/epoch
# This shows that just having more layers + backprop is not enough to learn addition, we need more tricks
# - Early stopping for full match of solution
# - Solution should probably 

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


def generate_ab(ndigits, allow_plus1=True, allow_less_digits=True):
    ndigits1 = ndigits
    ndigits2 = ndigits
    if allow_less_digits:
        ndigits1 = random.randint(1, ndigits)
        ndigits2 = random.randint(1, ndigits)
    while True:
        a = random.randint(10**(ndigits1-1), 10**ndigits1 - 1)
        b = random.randint(10**(ndigits2-1), 10**ndigits2 - 1)
        if a + b >= 10**ndigits and not allow_plus1:
            continue
        return a, b

def generate_ab_multiplication(ndigits, allow_less_digits=True):
    ndigits1 = ndigits
    ndigits2 = ndigits
    if allow_less_digits:
        ndigits1 = random.randint(1, ndigits)
        ndigits2 = random.randint(1, ndigits)
    while True:
        a = random.randint(10**(ndigits1-1), 10**ndigits1 - 1)
        b = random.randint(10**(ndigits2-1), 10**ndigits2 - 1)
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
    parser.add_argument('--addx', action='store_true', help='Add x input to each layer GRU')
    parser.add_argument('--allow-less-digits', action='store_true', help='Allow addends to use fewer digits than ndigits')
    parser.add_argument('--allow-plus1', action='store_true', help='Allow sums that overflow ndigits (one extra digit)')
    parser.add_argument('--mask', action='store_true', help='Use causal mask in transformer (off by default)')
    parser.add_argument('--mul', action='store_true', help='Generate multiplication dataset instead of addition')
    parser.add_argument('--first-char', action='store_true', help='Only keep first digit of result after "="')
    return parser.parse_args()

# --- Data Generation ---
class AdditionDataset(Dataset):
    def __init__(self, size, ndigits, vocab, device, allow_plus1=False, allow_less_digits=False, mask=False,
                 use_multiplication=False, first_char=False):
        self.size = size
        self.ndigits = ndigits
        self.vocab = vocab
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.device = device
        self.allow_plus1 = allow_plus1
        self.allow_less_digits = allow_less_digits
        self.use_multiplication = use_multiplication
        self.first_char = first_char
        self.operator_char = '*' if self.use_multiplication else '+'
        self.seq_len = self._compute_seq_len()
        self.data = self._generate_data()
        self.mask = mask

    def _compute_seq_len(self):
        if self.use_multiplication:
            max_result_digits = self.ndigits * 2
        else:
            max_result_digits = self.ndigits + 1
        return self.ndigits * 2 + max_result_digits + 2


    def _generate_data(self):
        rows = []
        seen = set()
        
        # Structure: "1234+5678=3579" (Fixed length)
        # Length = ndigits + 1 + ndigits + 1 + (ndigits) = 3 * ndigits + 2
        while len(rows) < self.size:
            if self.use_multiplication:
                a, b = generate_ab_multiplication(self.ndigits, allow_less_digits=self.allow_less_digits)
                res = a * b
            else:
                a, b = generate_ab(self.ndigits, allow_plus1=self.allow_plus1, allow_less_digits=self.allow_less_digits)
                res = a + b
            res_str = str(res)
            if self.first_char:
                res_str = res_str[:1]
            eqn = f"{a}{self.operator_char}{b}={res_str}"
            if len(eqn) > self.seq_len:
                continue
            eqn = eqn.ljust(self.seq_len, ' ')  # Pad with spaces if needed

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
        if self.mask:
            x = seq[:-1] # Inputs
            y = seq[1:]  # Targets
        else:
            x = seq.clone()
            # Find = in x and put spaces ater it
            equal_pos = (x == self.char_to_idx['=']).nonzero(as_tuple=True)[0].item()
            x[equal_pos+1:] = self.char_to_idx[' ']
            y = seq
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

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int, norm: nn.LayerNorm | None = None,
                 addx=True):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(copy.deepcopy(encoder_layer) for _ in range(num_layers))
        self.gru_cells = nn.ModuleList(
            nn.GRUCell(self.layers[0].d_model, self.layers[0].d_model) for _ in range(len(self.layers))
        )
        self.norm = norm
        self.addx = addx


    def forward(self, src, mask=None, src_key_padding_mask=None, x=None):
        output = src
        for idx, mod in enumerate(self.layers):
            layer_out = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            # GRU mixes previous layer output (hidden) with current layer output (input).
            b, s, d = layer_out.shape
            grux = layer_out.reshape(-1, d)
            if (x is not None) and self.addx:
                grux += x.reshape(-1, d).detach()
            output = self.gru_cells[idx](
                grux,
                output.reshape(-1, d),
            ).view(b, s, d)
        if self.norm is not None:
            output = self.norm(output)
        return output


class ToyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_heads, max_len, addx=False, use_mask=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.use_mask = use_mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1),
            persistent=False,
        )
        encoder_layer = MyTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = MyTransformerEncoder(encoder_layer, num_layers=n_layers, addx=addx)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def mask(self, sz):
        return self.causal_mask[:sz, :sz]

    def prepare_forward(self, x):
        # x shape: [Batch, SeqLen]
        # Causal Mask: Upper triangular is -inf
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        return emb

    def forward(self, x):
        emb, mask = self.prepare_forward(x), (self.mask(x.size(1)) if self.use_mask else None)
        # Transformer expects [Batch, Seq, Dim] because we set batch_first=True
        out = self.transformer(emb, mask=mask)
        out = out + emb  # retain top-level skip from embeddings to logits
        logits = self.fc_out(out)
        return logits

# --- Training & Evaluation ---

def evaluate_model(model, data, seq_len, ndigits, batch_size, mask=False, n_olayers=1):
    model.eval()
    correct_eq = 0
    correct_chars = 0
    total_chars = 0
    prompt_len = seq_len - ndigits

    if mask:
        x_all = data[:, :-1]
        y_all = data[:, 1:]
    else:
        x_all = data
        y_all = data
        # Find = in x and put spaces ater it
        equal_pos = (x_all == model.embedding.num_embeddings - 3).nonzero(as_tuple=True)[1]
        for i in range(x_all.size(0)):
            x_all[i, equal_pos[i]+1:] = model.embedding.num_embeddings - 1  # space idx
    total = data.size(0)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            x = x_all[start:end]
            y = y_all[start:end]

            if mask:
                # Prompt is everything up to and including '='
                prompt = x[:, :prompt_len]
                generated = prompt

                for _ in range(ndigits):
                    out = model(generated)
                    next_tok = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)

                expected_full = torch.cat([x, y[:, -1:]], dim=1)
            else:
                #out = model(x)
                emb = model.prepare_forward(x)
                running_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
                out = model.fc_out(emb)
                for _ in range(0, n_olayers):
                    emb = model.transformer(emb, x=emb)
                    #out = model.fc_out(emb)
                    current_logits = model.fc_out(emb)
                    out[running_mask] = current_logits[running_mask]
                    preds = torch.argmax(current_logits, dim=2)
                    matched = preds == y
                    running_mask = running_mask & (~matched).any(dim=1)
                generated = torch.argmax(out, dim=-1)
                expected_full = y

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

    # Vocab: 0-9, +, =, space
    vocab = "0123456789+=* "
    
    # Dataset: generate once and split into train/test
    total_size = args.train_size + args.test_size
    full_dataset = AdditionDataset(
        total_size,
        args.ndigits,
        vocab,
        device,
        allow_plus1=args.allow_plus1,
        allow_less_digits=args.allow_less_digits,
        mask=args.mask,
        use_multiplication=args.mul,
        first_char=args.first_char,
    )
    # Print some examples
    
    seq_len = full_dataset.seq_len
    train_dataset, test_dataset = random_split(
        full_dataset,
        [args.train_size, args.test_size],
        generator=torch.Generator().manual_seed(42),
    )
    eq_idx = full_dataset.char_to_idx['=']
    space_idx = full_dataset.char_to_idx[' ']
    train_indices = torch.tensor(train_dataset.indices, device=device, dtype=torch.long)
    test_indices = torch.tensor(test_dataset.indices, device=device, dtype=torch.long)
    train_data = full_dataset.data.index_select(0, train_indices)
    test_data = full_dataset.data.index_select(0, test_indices)
    if args.mask:
        train_inputs = train_data[:, :-1]
        train_targets = train_data[:, 1:]
        test_inputs = test_data[:, :-1]
        test_targets = test_data[:, 1:]
    else:
        train_targets = train_data
        test_targets = test_data
        # Find = in x and put spaces ater it
        equal_pos = (train_data == full_dataset.char_to_idx['=']).nonzero(as_tuple=True)[1]
        train_inputs = train_data.clone()
        for i in range(train_inputs.size(0)):
            train_inputs[i, equal_pos[i]+1:] = full_dataset.char_to_idx[' ']
        equal_pos = (test_data == full_dataset.char_to_idx['=']).nonzero(as_tuple=True)[1]
        test_inputs = test_data.clone()
        for i in range(test_inputs.size(0)):
            test_inputs[i, equal_pos[i]+1:] = full_dataset.char_to_idx[' ']
    
    print("Some example data:")
    for i in range(3):
        x, y = train_inputs[i], train_targets[i]
        x_str = ''.join([vocab[idx.item()] for idx in x])
        y_str = ''.join([vocab[idx.item()] for idx in y])
        print(f"Input: {x_str} | Target: {y_str}")
    # Model
    model = ToyLLM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=full_dataset.seq_len,  # For Positional Encoding
        addx=args.addx,
        use_mask=args.mask,
    ).to(device)
    mask = model.mask(seq_len-1).to(device) if model.use_mask else None
    
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
    effective_batch_size = min(args.batch_size, train_inputs.size(0))
    # Each epoch runs enough truncated steps so every example sees roughly n_olayers passes.
    steps_per_epoch = math.ceil(train_inputs.size(0) / effective_batch_size) * args.n_olayers
    # Persistent per-slot state
    step = torch.zeros(effective_batch_size, device=device, dtype=torch.long)
    current_x = torch.empty(effective_batch_size, train_inputs.size(1), dtype=torch.long, device=device)
    current_x_embed = torch.empty(effective_batch_size, train_inputs.size(1), args.embed_dim, device=device)
    current_y = torch.empty_like(current_x)
    hidden = torch.zeros(effective_batch_size, train_inputs.size(1), args.embed_dim, device=device)
    token_positions = torch.arange(train_inputs.size(1), device=device)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        total_steps_to_halt = 0.0
        total_halts = 0
        results_match_halts = 0
        step.zero_()
        # Shuffle the pool of training examples each epoch
        perm = torch.randperm(train_inputs.size(0), device=device)
        perm_cursor = 0

        def _draw_indices(k):
            nonlocal perm, perm_cursor
            if k == 0:
                return perm[:0]
            if perm_cursor + k > train_inputs.size(0):
                perm = torch.randperm(train_inputs.size(0), device=device)
                perm_cursor = 0
            idx = perm[perm_cursor : perm_cursor + k]
            perm_cursor += k
            return idx
        
        # Initial fill
        reset_mask = torch.ones(effective_batch_size, device=device, dtype=torch.bool)
        if reset_mask.any():
            idx = _draw_indices(reset_mask.sum().item())
            current_x[reset_mask] = train_inputs[idx]
            current_y[reset_mask] = train_targets[idx]
            current_x_embed[reset_mask] = model.prepare_forward(current_x[reset_mask])
            hidden[reset_mask] = model.prepare_forward(current_x[reset_mask])
            #hidden[reset_mask] = current_x_embed[reset_mask]


        for _ in range(steps_per_epoch):
            reset_mask = step == 0
            if reset_mask.any():
                idx = _draw_indices(reset_mask.sum().item())
                current_x[reset_mask] = train_inputs[idx]
                current_x_embed[reset_mask] = model.prepare_forward(current_x[reset_mask])
                #hidden[reset_mask] = current_x_embed[reset_mask]
                current_y[reset_mask] = train_targets[idx]
                hidden[reset_mask] = model.prepare_forward(current_x[reset_mask])


            # Single truncated step; gradients do not flow across steps because we detach below.
            for opt in optimizers:
                opt.zero_grad()
            
            hidden = model.transformer(hidden, mask=mask, x=current_x_embed)
            output = model.fc_out(hidden)
            
            # Output: [Batch, SeqLen, Vocab]
            # Target: [Batch, SeqLen]
            # Flatten for Loss
            loss = criterion(output.reshape(-1, len(vocab)), current_y.reshape(-1))
            
            loss.backward()
            for opt in optimizers:
                opt.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            with torch.no_grad():
                preds = output.argmax(dim=-1)
                match_mask = (preds == current_y)
                result_match = match_mask.all(dim=1)
                step = step + 1
                halt =  (step >= args.n_olayers) | result_match
                # Track how many steps each halted example needed this epoch.
                halted_steps = step[halt]
                total_steps_to_halt += halted_steps.sum().item()
                total_halts += halt.sum().item()
                results_match_halts += result_match.sum().item()
                step = torch.where(halt, torch.zeros_like(step), step)
                hidden = hidden.detach()
                hidden[halt] = 0.0  # placeholder state until we refill next loop
        
        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        avg_steps_to_halt = total_steps_to_halt / max(total_halts, 1)
        avg_exact_match_rate = results_match_halts / max(total_halts, 1)

        # Evaluation: test loss
        model.eval()
        total_test_loss = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for start in range(0, test_inputs.size(0), args.batch_size):
                end = min(start + args.batch_size, test_inputs.size(0))
                x = test_inputs[start:end]
                y = test_targets[start:end]
                xemb = model.prepare_forward(x)
                emb = xemb
                for _ in range(0, args.n_olayers):
                    emb = model.transformer(emb, mask=mask, x=xemb)
                out = model.fc_out(emb)
                test_loss = criterion(out.reshape(-1, len(vocab)), y.reshape(-1))
                total_test_loss += test_loss.item()
                num_test_batches += 1
        avg_test_loss = total_test_loss / max(num_test_batches, 1)

        # Evaluation: generation-based accuracies (train and test)
        train_char_acc, train_total_acc = evaluate_model(
            model, train_data, seq_len, args.ndigits, args.batch_size, mask=args.mask, n_olayers=args.n_olayers
        )
        test_char_acc, test_total_acc = evaluate_model(
            model, test_data, seq_len, args.ndigits, args.batch_size, mask=args.mask, n_olayers=args.n_olayers
        )

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{args.epochs}, "
            f"Train Time: {epoch_time:.2f}s, "
            f"loss: {avg_train_loss:.4f}, "
            f"char Acc: {train_char_acc*100:.2f}%, "
            f"exact acc: {train_total_acc*100:.2f}%, "
            f"Test Loss: {avg_test_loss:.4f}, "
            f"char acc: {test_char_acc*100:.2f}%, "
            f"exact acc: {test_total_acc*100:.2f}%, "
            f"Avg Steps: {avg_steps_to_halt:.2f}, "
            f"During training exact Match Rate: {avg_exact_match_rate*100:.2f}%"
        )
    
if __name__ == "__main__":
    args = get_args()
    train(args)

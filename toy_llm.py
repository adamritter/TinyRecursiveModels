import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for transformer feedforward')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    return parser.parse_args()

# --- Data Generation ---
class AdditionDataset(Dataset):
    def __init__(self, size, ndigits, vocab):
        self.size = size
        self.ndigits = ndigits
        self.vocab = vocab
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.data = self._generate_data()

    def _generate_data(self):
        dataset = []
        seen = set()
        
        # Structure: "1234+5678=3579" (Fixed length)
        # Length = ndigits + 1 + ndigits + 1 + (ndigits) = 3 * ndigits + 2
        
        self.seq_len = self.ndigits * 3 + 2
        
        while len(dataset) < self.size:
            a, b = generate_ab(self.ndigits)
            res = a + b
            eqn = f"{a}+{b}={res}"

            if eqn in seen:
                continue
            
            seen.add(eqn)
            
            # Convert to indices
            indices = [self.char_to_idx[c] for c in eqn]
            dataset.append(torch.tensor(indices, dtype=torch.long))
            
        return dataset

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x shape: [Batch, SeqLen]
        # Causal Mask: Upper triangular is -inf
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        emb = self.embedding(x)
        emb = self.pos_encoder(emb)

        # Transformer expects [Batch, Seq, Dim] because we set batch_first=True
        out = self.transformer(emb, mask=mask)
        logits = self.fc_out(out)
        return logits

# --- Training & Evaluation ---

def generate_equation(model, dataset, device, ndigits):
    model.eval()
    
    # Create a random test case: "1234+5678="
    a, b = generate_ab(ndigits)
    prompt_str = f"{a:0{ndigits}d}+{b:0{ndigits}d}="
    
    # Expected result
    expected_res = a + b
    expected_full = f"{prompt_str}{expected_res:0{ndigits}d}"
    
    # Convert prompt to tensor
    indices = [dataset.char_to_idx[c] for c in prompt_str]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device) # [1, SeqLen]
    
    print(f"\nPrompt: {prompt_str}")
    
    with torch.no_grad():
        for _ in range(ndigits): # Generate exactly ndigits characters
            output = model(input_tensor)
            
            # Get logits for the last token generated
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to input
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
    # Decode
    generated_indices = input_tensor[0].cpu().numpy()
    generated_str = "".join([dataset.idx_to_char[i] for i in generated_indices])
    
    is_correct = (generated_str == expected_full)
    print(f"Generated: {generated_str}")
    print(f"Expected:  {expected_full}")
    print(f"Correct:   {is_correct}")
    return is_correct

def evaluate_model(model, dataset, seq_len, ndigits, device, batch_size):
    model.eval()
    correct_eq = 0
    correct_chars = 0
    total_chars = 0
    prompt_len = seq_len - ndigits
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == 'cuda',
    )
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
    total_acc = correct_eq / len(dataset) if len(dataset) > 0 else 0.0
    return char_acc, total_acc

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Vocab: 0-9, +, =
    vocab = "0123456789+="
    
    # Dataset: generate once and split into train/test
    total_size = args.train_size + args.test_size
    full_dataset = AdditionDataset(total_size, args.ndigits, vocab)
    seq_len = full_dataset.seq_len
    pin_memory = torch.cuda.is_available()
    train_dataset, test_dataset = random_split(
        full_dataset,
        [args.train_size, args.test_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(x)
            
            # Output: [Batch, SeqLen, Vocab]
            # Target: [Batch, SeqLen]
            # Flatten for Loss
            loss = criterion(output.reshape(-1, len(vocab)), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation: test loss
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                test_loss = criterion(out.reshape(-1, len(vocab)), y.reshape(-1))
                total_test_loss += test_loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        # Evaluation: generation-based accuracies (train and test)
        train_char_acc, train_total_acc = evaluate_model(
            model, train_dataset, seq_len, args.ndigits, device, args.batch_size
        )
        test_char_acc, test_total_acc = evaluate_model(
            model, test_dataset, seq_len, args.ndigits, device, args.batch_size
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

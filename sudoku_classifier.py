import random
import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import numpy as np

base = 3
side = base * base


def pattern(r, c):
    return (base * (r % base) + r // base + c) % side


def shuffle(s):
    return random.sample(s, len(s))


def make_board():
    rBase = range(base)
    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, side + 1))
    return [[nums[pattern(r, c)] for c in cols] for r in rows]


def clone_board(b):
    return [row[:] for row in b]


def modify_random_cell(b):
    r = random.randint(0, side - 1)
    c = random.randint(0, side - 1)
    v = random.randint(1, 8)
    if v < b[r][c]:
        b[r][c] = v
    else:
        b[r][c] = v + 1
    return b

def random_sudoku_board():
    return [[random.randint(1, 9) for _ in range(9)] for _ in range(9)]


def swap_in_row(b):
    r = random.randint(0, side - 1)
    c1, c2 = random.sample(range(side), 2)
    if c1 == c2:
        return swap_in_row(b)
    b[r][c1], b[r][c2] = b[r][c2], b[r][c1]
    return b


def swap_in_column(b):
    c = random.randint(0, side - 1)
    r1, r2 = random.sample(range(side), 2)
    if r1 == r2:
        return swap_in_column(b)
    b[r1][c], b[r2][c] = b[r2][c], b[r1][c]
    return b


#def bad_board_modification(b):
#    modification = random.choice([modify_random_cell, swap_in_row, swap_in_column])
#    return modification(b)


def print_board(b):
    for r in range(side):
        for c in range(side):
            print(b[r][c], end=" ")
        print()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS Metal backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_one_hot_flat(board_array, minv: int = None, maxv: int = None):
    """One-hot encode flat label arrays using PyTorch.

    - Accepts shape (N, L) or (L,) with integer values.
    - Uses value range [minv, maxv] mapped to indices [0, C-1], where C = maxv-minv+1.
    - Returns FloatTensor of shape (N, L*C) on CPU.
    """
    t = torch.as_tensor(board_array, dtype=torch.long)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    else:
        t = t.reshape(t.size(0), -1)
    if minv is None:
        minv = int(t.min().item())
    if maxv is None:
        maxv = int(t.max().item())
    C = maxv - minv + 1
    idx = t - minv
    if idx.min().item() < 0 or idx.max().item() >= C:
        raise ValueError("Values out of expected range after offset. Check minv/maxv or inputs.")
    one_hot = torch.nn.functional.one_hot(idx, num_classes=C).to(dtype=torch.float32)
    one_hot = one_hot.reshape(t.size(0), -1)
    return one_hot


def test_model(model, X_test, y_test):
    device = get_device()
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
    acc = correct / total
    return acc


def train_model(X_train, y_train, X_test, y_test, save="sudoku_classifier.pth"):
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    device = get_device()

    input_dim = int(X_train.shape[1])
    model = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        test_acc = test_model(model, X_test, y_test)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | test_acc={test_acc:.4f}")     
    print(f"Final test accuracy: {test_acc:.4f}")
    torch.save(model.state_dict(), save)
    return model


class FlatSudokuDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that streams labels and generates negatives on-the-fly.

    - labels_path: path to .npy file with shape (N, L) of integer tokens.
    - minv/maxv: inclusive value range used for negatives and one-hot.
    - include_random_neg: if True, the dataset length is 2*N (positives + random negatives).
    - seed: base seed to deterministically generate negatives per index.
    """
    def __init__(self, labels_path: str, minv: int = None, maxv: int = None,
                 include_random_neg: bool = True, seed: int = 42):
        self.labels_np = np.load(labels_path, mmap_mode='r')
        if self.labels_np.ndim != 2:
            self.labels_np = self.labels_np.reshape(self.labels_np.shape[0], -1)
        self.N, self.L = int(self.labels_np.shape[0]), int(self.labels_np.shape[1])
        self.minv = int(self.labels_np.min()) if minv is None else int(minv)
        self.maxv = int(self.labels_np.max()) if maxv is None else int(maxv)
        self.C = self.maxv - self.minv + 1
        self.include_random_neg = include_random_neg
        self.seed = int(seed)

    def __len__(self):
        return self.N * 2 if self.include_random_neg else self.N

    def __getitem__(self, idx):
        if idx < self.N:
            x = np.array(self.labels_np[idx], dtype=np.int64)
            y = 1.0
        else:
            ridx = idx - self.N
            rng = np.random.default_rng(self.seed + ridx)
            x = rng.integers(self.minv, self.maxv + 1, size=self.L, dtype=np.int64)
            y = 0.0
        return torch.as_tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


def test_model_streaming(model: nn.Module, ds: torch.utils.data.Dataset, batch_size: int = 512) -> float:
    device = get_device()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            # One-hot per batch on device
            C = getattr(ds, 'C', int(tokens.max().item() - tokens.min().item() + 1))
            minv = getattr(ds, 'minv', int(tokens.min().item()))
            idx = tokens - minv
            one_hot = F.one_hot(idx, num_classes=C).to(torch.float32)
            x = one_hot.view(tokens.size(0), -1)
            logits = model(x).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).to(labels.dtype)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train_model_streaming(train_ds: torch.utils.data.Dataset,
                          test_ds: torch.utils.data.Dataset,
                          save: str = "sudoku_classifier.pth",
                          batch_size: int = 512,
                          epochs: int = 10,
                          lr: float = 1e-4):
    device = get_device()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Infer input dimension from dataset metadata
    L = getattr(train_ds, 'L')
    C = getattr(train_ds, 'C')
    minv = getattr(train_ds, 'minv')
    input_dim = L * C

    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            idx = tokens - minv
            one_hot = F.one_hot(idx, num_classes=C).to(torch.float32)
            x = one_hot.view(tokens.size(0), -1)
            logits = model(x).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bsz = tokens.size(0)
            running += loss.item() * bsz
            seen += bsz
        avg_loss = running / max(seen, 1)

        # Eval
        acc = test_model_streaming(model, test_ds, batch_size=batch_size)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | test_acc={acc:.4f}")

    torch.save(model.state_dict(), save)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None, help="Path containing train/ and test/ folders")
    parser.add_argument("--save", type=str, default="sudoku_classifier.pth", help="Path to save model weights")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.datapath is not None:
        def path_labels(split: str):
            cand1 = os.path.join(args.datapath, split, "all_labels.npy")
            cand2 = os.path.join(args.datapath, split, "all__labels.npy")
            if os.path.exists(cand1):
                return cand1
            if os.path.exists(cand2):
                return cand2
            raise FileNotFoundError(f"Could not find labels at {cand1} or {cand2}")

        print("Loading dataset from:", args.datapath)
        train_path = path_labels("train")
        test_path = path_labels("test")
        train_ds = FlatSudokuDataset(train_path, include_random_neg=True, seed=42)
        test_ds = FlatSudokuDataset(test_path, include_random_neg=True, minv=train_ds.minv, maxv=train_ds.maxv, seed=123)
        print(f"Train size: {len(train_ds)} (pos+neg), Test size: {len(test_ds)} (pos+neg). L={train_ds.L}, C={train_ds.C}")
        model = train_model_streaming(train_ds, test_ds, save=args.save, batch_size=512, epochs=10, lr=1e-4)
    else:
        # Generate synthetic dataset
        n = 100000
        boards = [make_board() for _ in range(n)]
        bad_boards = [random_sudoku_board() for _ in range(n // 2)] + [modify_random_cell(make_board()) for _ in range(n // 2)]

        X_good = encode_one_hot_flat(boards)
        X_bad = encode_one_hot_flat(bad_boards)

        y_good = torch.ones(X_good.shape[0], dtype=torch.float32)
        y_bad = torch.zeros(X_bad.shape[0], dtype=torch.float32)

        X = torch.cat([X_good, X_bad], dim=0)
        y = torch.cat([y_good, y_bad], dim=0)

        n_good = len(boards)
        idx_good_train = slice(0, int(0.8 * n_good))
        idx_good_test = slice(int(0.8 * n_good), n_good)
        idx_bad_train = slice(n_good, n_good + int(0.8 * n_good))
        idx_bad_test = slice(n_good + int(0.8 * n_good), 2 * n_good)

        train_indices = torch.cat([torch.arange(idx_good_train.start, idx_good_train.stop),
                                   torch.arange(idx_bad_train.start, idx_bad_train.stop)])
        test_indices = torch.cat([torch.arange(idx_good_test.start, idx_good_test.stop),
                                  torch.arange(idx_bad_test.start, idx_bad_test.stop)])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        model = train_model(X_train, y_train, X_test, y_test, save=args.save)

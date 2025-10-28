import random
import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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


def modify_random_cell_batch(boards: torch.Tensor, minv: int = None, maxv: int = None, generator: torch.Generator = None):
    """Vectorized variant of modify_random_cell that perturbs one entry per board."""
    if boards.dim() < 2:
        raise ValueError("Expected `boards` to have batch dimension (N, ...).")

    clone = boards.clone()
    batch = clone.shape[0]
    flat = clone.view(batch, -1)

    if minv is None:
        minv = int(flat.min().item())
    if maxv is None:
        maxv = int(flat.max().item())
    num_vals = maxv - minv + 1
    if num_vals < 2:
        raise ValueError("Need at least two possible values to modify a cell.")

    device = flat.device
    idx = torch.randint(0, flat.size(1), (batch,), generator=generator, device=device)
    row_ids = torch.arange(batch, device=device)
    current = flat[row_ids, idx]

    offsets = torch.randint(0, num_vals - 1, (batch,), generator=generator, device=device)
    current_offset = current - minv
    new_offset = offsets + (offsets >= current_offset).to(offsets.dtype)
    new_values = (new_offset + minv).to(flat.dtype)

    flat[row_ids, idx] = new_values
    return clone.view_as(boards)

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


def build_classifier(input_dim, device=None):
    if device is None:
        device = get_device()
    model = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    return model.to(device)


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


def train_model(X_train, y_train, X_test, y_test, save="sudoku_classifier.pth", epochs=50):
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    device = get_device()

    input_dim = int(X_train.shape[1])
    model = build_classifier(input_dim, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None, help="Path containing train/ and test/ folders")
    parser.add_argument("--save", type=str, default="sudoku_classifier.pth", help="Path to save model weights")
    parser.add_argument("--eval", type=str, default=None, help="Path to model weights to evaluate instead of training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.datapath is not None:
        def load_labels(split: str):
            cand1 = os.path.join(args.datapath, split, "all_labels.npy")
            cand2 = os.path.join(args.datapath, split, "all__labels.npy")
            if os.path.exists(cand1):
                return torch.from_numpy(np.load(cand1))
            if os.path.exists(cand2):
                return torch.from_numpy(np.load(cand2))
            raise FileNotFoundError(f"Could not find labels at {cand1} or {cand2}")

        # Load solved sequences as 'good' examples (kept flat)
        print("Loading dataset from:", args.datapath)
        train_labels = load_labels("train")  # shape (N, L)
        test_labels = load_labels("test")    # shape (M, L)
        print(f"Train labels shape: {train_labels.shape}, Test labels shape: {test_labels.shape}")

        # Determine value range from training set for consistent encoding
        minv = int(train_labels.min())
        maxv = int(train_labels.max())

        # Create random 'bad' examples that match shape and value range
        rng = torch.Generator().manual_seed(42)
        train_bad = torch.randint(
            low=minv,
            high=maxv + 1,
            size=train_labels.shape,
            generator=rng,
            device=train_labels.device,
            dtype=torch.int64,
        )
        test_bad = modify_random_cell_batch(
            test_labels,
            minv=minv,
            maxv=maxv,
            generator=rng,
        )

        # print a test_bad example
        print("Example bad board (test set):")
        print(test_bad[0].view(9, 9))

        print(f"Value range for encoding: min={minv}, max={maxv}")

        X_test = torch.cat([
            encode_one_hot_flat(test_labels, minv=minv, maxv=maxv),
            encode_one_hot_flat(test_bad, minv=minv, maxv=maxv)
        ], dim=0)
        y_test = torch.cat([
            torch.ones(test_labels.shape[0], dtype=torch.float32),
            torch.zeros(test_bad.shape[0], dtype=torch.float32)
        ], dim=0)

        if args.eval:
            if not os.path.exists(args.eval):
                raise FileNotFoundError(f"Evaluation weights not found at: {args.eval}")
            device = get_device()
            model = build_classifier(int(X_test.shape[1]), device=device)
            state = torch.load(args.eval, map_location=device)
            model.load_state_dict(state)
            acc = test_model(model, X_test, y_test)
            print(f"Evaluation accuracy: {acc:.4f}")
        else:
            X_train = torch.cat([
                encode_one_hot_flat(train_labels, minv=minv, maxv=maxv),
                encode_one_hot_flat(train_bad, minv=minv, maxv=maxv)
            ], dim=0)
            y_train = torch.cat([
                torch.ones(train_labels.shape[0], dtype=torch.float32),
                torch.zeros(train_bad.shape[0], dtype=torch.float32)
            ], dim=0)

            print("Generating test dataset...")
            print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
            model = train_model(X_train, y_train, X_test, y_test, save=args.save, epochs=args.epochs)
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

        if args.eval:
            if not os.path.exists(args.eval):
                raise FileNotFoundError(f"Evaluation weights not found at: {args.eval}")
            device = get_device()
            model = build_classifier(int(X_train.shape[1]), device=device)
            state = torch.load(args.eval, map_location=device)
            model.load_state_dict(state)
            acc = test_model(model, X_test, y_test)
            print(f"Evaluation accuracy: {acc:.4f}")
        else:
            model = train_model(X_train, y_train, X_test, y_test, save=args.save, epochs=args.epochs)

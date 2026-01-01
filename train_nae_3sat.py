# Test for NAE_3SAT instance generation and training directly with Muon optimizer

import torch, math, time
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

def randbool_noeq(shape, dim=-1, device=None, generator=None) -> torch.Tensor:
    """
    Generates perfectly uniform random booleans skipping 'all-equal' states.
    Uses bit-unpacking for O(1) performance without loops.
    """
    n = shape[dim]
    num_valid_states = (2**n) - 1
    reduced_shape = list(shape)
    reduced_shape[dim] = 1
    ints = torch.randint(1, num_valid_states, reduced_shape, device=device, generator=generator)
    bits = [(ints >> i) & 1 for i in range(n)]
    return torch.cat(bits, dim=dim).to(torch.bool)

def sample_of_n(n: int, size: torch.Size, *, device=None, generator=None) -> torch.Tensor:
    """
    Sample `size[-1]` distinct integers uniformly from {0,1,...,n-1},
    independently for every other batch index (all leading dims).

    Example: size=(B, 3) -> 3 unique per batch row.
             size=(B, T, K) -> K unique per (B,T) position.
    """
    k = size[-1]
    if k <= 0 or k > n:
        raise ValueError("size[-1] must be > 0 and <= n")
    sel = torch.empty(size, dtype=torch.long, device=device)
    for i in range(k):
        r = torch.randint(n - i, size[:-1], device=device, generator=generator)
        x = r
        for _ in range(i):  # fixed-point; converges fast for small k (your k=3 case)
            x = r + (sel[..., :i] <= x.unsqueeze(-1)).sum(dim=-1)
        sel[..., i] = x
    return sel

def nae_3sat(n, clause_multiple=2.11, batch_size=None, device=None, generator=None):
    """
    Generates random 3-NAE-SAT instances with guaranteed satisfiability.
    Each clause has exactly 3 literals, and no-all-equal (NAE) condition (2 clauses per example)

    Args:
        n: Number of variables.
        clause_multiple: Number of clauses = clause_multiple * 2 * n.
        batch_size: If provided, generates a batch of instances.
        device: Torch device.

    Returns:
        problems: Tensor of shape (num_clauses, 3) or (batch_size, num_clauses, 3)
                  containing DIMACS-signed literals (1-indexed, negative for negated).
        assignment: Tensor of shape (n,) or (batch_size, n) with boolean assignments.
    """
    if n < 3:
        raise ValueError("Need n >= 3 for 3 distinct variables per clause.")
    B = 1 if batch_size is None else batch_size
    num_clauses = int(clause_multiple * n)
    assignment = torch.randint(0, 2, (B, n), device=device, generator=generator).bool()
    clauses = sample_of_n(n, (B, num_clauses, 3), device=device, generator=generator)
    target = randbool_noeq(clauses.shape, dim=-1, device=device, generator=generator)
    negated = target ^ assignment.gather(1, clauses.reshape(B, -1)).reshape(B, num_clauses, 3)
    # Duplicate each clause once, pairing it with both its negation pattern and its inverse
    clauses = clauses.repeat(1, 2, 1)                # (B, 2*num_clauses, 3)
    negated = torch.cat([negated, ~negated], dim=1)  # (B, 2*num_clauses, 3)
    # Create problem (vars indexed from 1 in DIMACS format + plus/minus for negation)
    problems = clauses + 1
    problems[negated] *= -1
    print("n=", n, "multiple=", clause_multiple, "problems=", problems.size())
    if batch_size is None:
        return problems[0], assignment[0]
    return problems, assignment

def write_cnf(problems: torch.Tensor, filepath: str):
    """
    Writes a CNF file in DIMACS format.

    Args:
        problems: Tensor of shape (num_clauses, k) with DIMACS-signed literals (1-indexed).
        filepath: Output file path.
    """
    num_vars = problems.abs().max().item()
    num_clauses = problems.size(0)

    with open(filepath, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in problems.tolist():
            f.write(" ".join(map(str, clause)) + " 0\n")

def print_assignment(assignment: torch.Tensor):
    """
    Print a planted assignment with indices starting at 0 using DIMACS sign convention.
    """
    print("Planted assignment:", end=' ')
    for i, val in enumerate(assignment):
        print(-i - 1 if val.item() else i + 1, end=' ')
    print()

_LOG_HALF = -math.log(2.0)  # log(0.5)
def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(1 - exp(x)) for x <= 0.
    Works elementwise, supports autograd, CPU/GPU.
    """
    x = torch.as_tensor(x)
    x = x.clamp_max(0)  # safety: logsigmoid sums should be <= 0

    return torch.where(
        x > _LOG_HALF,               # x in (-log 2, 0]
        torch.log(-torch.expm1(x)),  # stable when exp(x) ~ 1
        torch.log1p(-torch.exp(x)),  # stable when exp(x) is small
    )

def prod_logits(logits: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Given independent Bernoulli logits, compute the logit of the product prob:
        p = ∏ sigmoid(logits_i)
        return logit(p) = log(p) - log(1-p)
    in a stable log-domain way.
    """
    if logits.size(dim) == 0:
        raise ValueError(f"prod_logits: empty dimension dim={dim} (product would be 1 -> logit=+inf).")

    log_p = F.logsigmoid(logits).sum(dim=dim, keepdim=keepdim).clamp_max(0)
    return log_p - log1mexp(log_p)

def prod_logits_logprob(logits: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Given independent Bernoulli logits, compute the log probability of the product:
        p = ∏ sigmoid(logits_i)
    in a stable log-domain way.
    """
    if logits.size(dim) == 0:
        raise ValueError(f"prod_logits_logprob: empty dimension dim={dim} (product would be 1 -> logit=+inf).")

    return F.logsigmoid(logits).sum(dim=dim, keepdim=keepdim).clamp_max(0)

def compute_clause_and_exact_accuracy(problems: torch.Tensor, var_logits: torch.Tensor):
    """
    Compute clause-level and exact problem accuracy for logits over variables.
    """
    if problems.ndim == 2:
        problems = problems.unsqueeze(0)
    if var_logits.ndim == 1:
        var_logits = var_logits.unsqueeze(0)

    B, num_clauses, _ = problems.shape
    clauses_abs = problems.abs().long() - 1
    clauses_sign = problems.sign()

    gathered_var_logits = var_logits.gather(1, clauses_abs.view(B, -1)).view(B, num_clauses, 3)
    lits_logits = clauses_sign * gathered_var_logits
    clause_satisfied = (lits_logits > 0).any(dim=-1)
    clause_accuracy = 100.0 * clause_satisfied.sum().item() / clause_satisfied.numel()
    exact_accuracy = 100.0 * clause_satisfied.all(dim=-1).sum().item() / B
    return clause_accuracy, exact_accuracy

def train_clauses(n, problems, steps=10, lr=100.0, generator=None):
    if problems.ndim == 2:
        problems = problems.unsqueeze(0)
    B, num_clauses, _ = problems.shape
    var_logits = torch.nn.Parameter(torch.randn(B, n, device=problems.device, generator=generator))
    t=time.time()
    clauses = problems.abs().long() - 1
    sign = problems.sign()
    opt = optim.Muon([var_logits], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        gathered = var_logits.gather(1, clauses.view(B, -1)).view(B, num_clauses, 3)
        lits_logits = sign * gathered
        clause_logprobs = F.logsigmoid(-prod_logits(-lits_logits, dim=-1))
        #total_loss = -clause_logprobs.sum()
        total_loss = torch.logsumexp(-clause_logprobs, dim=0).mean() # alternative: log-sum-exp of clause losses
        total_loss.backward()
        opt.step()

    with torch.no_grad():
        clause_accuracy, exact_accuracy = compute_clause_and_exact_accuracy(problems, var_logits)
    print(f"Step {step+1}: loss={total_loss.item():.6f}, clause_accuracy={clause_accuracy:.2f}%, exact_accuracy={exact_accuracy:.2f}% in {time.time()-t:.2f} seconds")

@dataclass
class GNNState:
    literals: torch.Tensor
    clauses: torch.Tensor

    def detach(self) -> "GNNState":
        return GNNState(literals=self.literals.detach(), clauses=self.clauses.detach())

class GNN(torch.nn.Module):
    def __init__(self, hidden_dim, num_rounds):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds

        # Learnable initial embeddings for literals and clauses
        # Separate positive/negative literal seeds to avoid symmetric collapse on tiny graphs
        self.literal_init_emb = torch.nn.Parameter(torch.randn(2, hidden_dim))
        self.clause_init_emb = torch.nn.Parameter(torch.randn(hidden_dim))

        # Message passing layers
        self.W_cl = torch.nn.Linear(hidden_dim, hidden_dim) # Clause to Literal
        self.W_lc = torch.nn.Linear(hidden_dim, hidden_dim) # Literal to Clause
        self.W_flip = torch.nn.Linear(hidden_dim, hidden_dim) # Negated literal

        # GRU cells for updates
        self.gru_literal = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_clause = torch.nn.GRUCell(hidden_dim, hidden_dim)

        # Readout head for variable logits
        self.readout = torch.nn.Linear(hidden_dim, 1)

    def init_state(self, problems):
        B, num_clauses, _ = problems.shape
        n = problems.abs().max().item()
        num_literals = 2 * n
        literal_signs = torch.arange(num_literals, device=problems.device) % 2
        literals = self.literal_init_emb[literal_signs].unsqueeze(0).expand(B, num_literals, -1).reshape(B * num_literals, self.hidden_dim).clone()
        clauses = self.clause_init_emb.unsqueeze(0).expand(B * num_clauses, -1).clone()
        return GNNState(literals=literals, clauses=clauses)

    def forward(self, problems, state: GNNState = None):
        B, num_clauses, _ = problems.shape
        n = problems.abs().max().item() # Number of variables
        num_literals = 2 * n
        state = self.init_state(problems) if state is None else state

        # Create graph structure for batch
        # Ci: clause index for each literal occurrence
        # Lj: literal index (0 to 2n-1) for each literal occurrence
        # flip_idx: index of the negated literal
        
        # Flatten problems for easier indexing
        flat_problems = problems.view(B, -1) # (B, num_clauses * 3)
        
        # Literal indices (0-indexed, 0 for var 1 pos, 1 for var 1 neg, etc.)
        # var_idx = (abs(literal) - 1) * 2
        # lit_idx = var_idx + (1 if literal < 0 else 0)
        Lj_flat = (flat_problems.abs() - 1) * 2 + (flat_problems < 0).long()
        
        # Clause indices (0-indexed)
        Ci_flat = torch.arange(num_clauses, device=problems.device).repeat_interleave(3).unsqueeze(0).expand(B, -1)

        # Batch offsets
        batch_clause_offset = torch.arange(B, device=problems.device) * num_clauses
        batch_literal_offset = torch.arange(B, device=problems.device) * num_literals

        Ci = (Ci_flat + batch_clause_offset.unsqueeze(-1)).view(-1)
        Lj = (Lj_flat + batch_literal_offset.unsqueeze(-1)).view(-1)

        # Flip indices for each literal (for each problem in batch)
        # For literal 2*v, its negation is 2*v+1
        # For literal 2*v+1, its negation is 2*v
        flip_idx_base = torch.arange(num_literals, device=problems.device)
        flip_idx_base = flip_idx_base + (1 - 2 * (flip_idx_base % 2))
        flip_idx = (flip_idx_base.unsqueeze(0) + batch_literal_offset.unsqueeze(-1)).view(-1)

        for _ in range(self.num_rounds):
            # Clause to Literal message passing
            msg_c2l = self.W_cl(state.clauses)
            agg_c2l = torch.zeros_like(state.literals)
            agg_c2l.index_add_(0, Lj, msg_c2l[Ci])

            # Negated literal message
            flip_in = state.literals[flip_idx]
            agg_flip = self.W_flip(flip_in)

            # Update literals
            lit_input = agg_c2l + agg_flip
            new_literals = self.gru_literal(lit_input, state.literals)

            # Literal to Clause message passing
            msg_l2c = self.W_lc(new_literals)
            agg_l2c = torch.zeros_like(state.clauses)
            agg_l2c.index_add_(0, Ci, msg_l2c[Lj])

            # Update clauses
            new_clauses = self.gru_clause(agg_l2c, state.clauses)
            state = GNNState(literals=new_literals, clauses=new_clauses)

        # Readout: get a score for each literal
        # Reshape Hl to (B, num_literals, hidden_dim)
        Hl_reshaped = state.literals.view(B, num_literals, self.hidden_dim)
        
        # The readout head gives a single score for each literal
        # We need to extract the scores for positive literals (even indices)
        # These scores will represent the logits for P(var=True)
        literal_scores = self.readout(Hl_reshaped).squeeze(-1) # (B, num_literals)
        var_logits = literal_scores[:, 0::2] # (B, n) - take scores for x1, x2, ..., xn

        return var_logits, state

def print_planted_assignment(n, assignment_batch_item):
    print("Planted assignment:", end=' ')
    for i in range(n):
        val = assignment_batch_item[i].item()
        print(-i - 1 if val else i + 1, end=' ')
    print()

def train_gnn(n, problems, steps=10, lr=1e-3, hidden_dim=16, num_rounds=10, generator=None, batch_size=None, test_size=None, outer_rounds=1):
    if problems.ndim == 2:
        problems = problems.unsqueeze(0)
    if test_size is not None:
        test = problems[0:test_size]
        problems = problems[test_size:]
    B, num_clauses, _ = problems.shape
    batch_size = B if batch_size is None else min(batch_size, B)

    model = GNN(hidden_dim, num_rounds).to(problems.device)
    
    # Separate parameters for Muon optimizer (2D tensors) and Adam (other tensors)
    params_2d = []
    params_other = []
    for p in model.parameters():
        if p.ndim == 2:
            params_2d.append(p)
        else:
            params_other.append(p)
    opt = optim.Adam(model.parameters(), lr=lr) # Fallback to Adam for simplicity, or use multiple optimizers

    t_start = time.time()

    last_loss = 0.0
    for step in range(steps):
        perm = torch.randperm(B, device=problems.device, generator=generator)
        total_loss_accum = 0.0
        total_seen = 0

        for start in range(0, B, batch_size):
            idx = perm[start:start + batch_size]
            batch = problems[idx]
            bB = batch.size(0)

            state = None
            for _ in range(outer_rounds):
                opt.zero_grad()
                var_logits, state = model(batch, state) # (bB, n)
                state = state.detach()

                # Same loss calculation as in train_clauses
                clauses_abs = batch.abs().long() - 1 # 0-indexed variable IDs
                clauses_sign = batch.sign()

                # Gather logits for each literal in each clause
                # For a literal `v`, if it's positive, we use var_logits[:, v-1]
                # If it's negative, we use -var_logits[:, v-1]
                # This is equivalent to: sign * var_logits.gather(1, abs(literal)-1)
                gathered_var_logits = var_logits.gather(1, clauses_abs.view(bB, -1)).view(bB, num_clauses, 3)
                lits_logits = clauses_sign * gathered_var_logits

                # Calculate clause satisfaction probabilities
                # P(clause_i is true) = 1 - P(all literals in clause_i are false)
                # P(literal_j is false) = sigmoid(-logit_j)
                # P(all literals in clause_i are false) = product(sigmoid(-logit_j))
                clause_logprobs = F.logsigmoid(-prod_logits(-lits_logits, dim=-1))
                total_loss = torch.logsumexp(-clause_logprobs, dim=0).mean()
                
                total_loss.backward()
                opt.step()

                total_loss_accum += total_loss.item() * bB
                total_seen += bB

        last_loss = total_loss_accum / max(total_seen, 1)

    with torch.no_grad():
        state = None
        for _ in range(outer_rounds):
            var_logits, state = model(problems, state)
        clause_accuracy, exact_accuracy = compute_clause_and_exact_accuracy(problems, var_logits)
    print(f"GNN Step {step+1}: loss={last_loss:.6f}, clause_accuracy={clause_accuracy:.2f}%, exact_accuracy={exact_accuracy:.2f}% in {time.time()-t_start:.2f} seconds")

    if test_size is not None:
        with torch.no_grad():
            t = time.time()
            state = None
            for _ in range(outer_rounds):
                var_logits, state = model(test, state)
            clause_accuracy, exact_accuracy = compute_clause_and_exact_accuracy(test, var_logits)
        print(f"GNN Test: clause_accuracy={clause_accuracy:.2f}%, exact_accuracy={exact_accuracy:.2f}% in {time.time()-t:.2f} seconds")

if __name__ == "__main__":
    n = 20
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    generator=torch.Generator(device=device).manual_seed(0)
    nae100_problems, nae100_assignments = nae_3sat(n, device=device, batch_size=2*4096+256, generator=generator)
    write_cnf(nae100_problems[0], 'test_nae3sat.cnf')
    print_assignment(nae100_assignments[0])
    train_clauses(n, nae100_problems[0:256], steps=1000, lr=1, generator=generator)
    train_gnn(n, nae100_problems, steps=160, lr=0.001, hidden_dim=16, num_rounds=15, generator=generator, batch_size=1024, test_size=256, outer_rounds=2)

    nae1000_problem, nae1000_assignment = nae_3sat(1000, device=device, generator=generator)
    write_cnf(nae1000_problem, 'test_nae3sat_big.cnf')

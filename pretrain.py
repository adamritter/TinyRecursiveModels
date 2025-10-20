from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []
    # Evaluation behavior
    eval_partial_finish: bool = False  # If True, evaluate with per-sample early stopping via indexing

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    eval_only: bool = False  # If True, skip training and run a single evaluation pass after loading the checkpoint

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


train_batch_counter = 0
# Current per-batch step budget (we avoid mutating model.config.halt_max_steps)
effective_halt_max_steps_state = None

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # ----- Effective halt limit (tensor) -----
    global effective_halt_max_steps_state
    if effective_halt_max_steps_state is None:
        # Initialize once from model’s configured cap
        effective_halt_max_steps_state = int(train_state.model.model.config.halt_max_steps)
    eff_limit_t = torch.tensor(effective_halt_max_steps_state, device="cuda", dtype=torch.int32)

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry,
        batch=batch,
        return_keys=[],
        effective_halt_max_steps=eff_limit_t,  # <— new arg
    )

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            # ---- Dynamic effective halt_max_steps adjustment (no module mutation) ----
            # If an environment variable provides a maximum-step override and the
            # model is already halting accurately, expand the per-batch budget.
            max_step_override_env = (
                os.environ.get("MAX_STEP_OVERRIDE")
                or os.environ.get("HALT_MAX_STEPS_OVERRIDE")
            )
            if max_step_override_env is not None and "train/q_halt_accuracy" in reduced_metrics:
                try:
                    override_val = int(max_step_override_env)
                    global train_batch_counter
                    current_halt_max = int(effective_halt_max_steps_state)
                    train_batch_counter += 1
                    if (reduced_metrics["train/q_halt_accuracy"] > 0.95 and override_val > current_halt_max):
                        if train_batch_counter == 5:
                            effective_halt_max_steps_state = current_halt_max + 1
                            print(f"Auto-increased effective halt_max_steps to {effective_halt_max_steps_state} (no module mutation)")
                            train_batch_counter = 0
                    else:
                        train_batch_counter = 0
                except ValueError:
                    # Ignore malformed override values
                    pass
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    global effective_halt_max_steps_state
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)
        # Ensure q_halt_logits is available if partial-finish mode is enabled
        if config.eval_partial_finish:
            return_keys.add("q_halt_logits")

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            if effective_halt_max_steps_state is None:
                effective_halt_max_steps_state = int(train_state.model.model.config.halt_max_steps)
            eff_limit_t = torch.tensor(effective_halt_max_steps_state, device="cuda", dtype=torch.int32)
            # Implement partial-finish indexing mode using q_halt_logits > 0
            if config.eval_partial_finish:
                B = batch["inputs"].shape[0]
                global_alive = torch.arange(B, device="cuda")
                final_preds: dict = {}
                step_counts = torch.zeros(B, dtype=torch.int32, device="cuda")
                sub_batch = batch
                # Run up to the halt cap; on last step force-finish all remaining
                max_steps = int(effective_halt_max_steps_state)
                for t in range(max_steps):
                    if global_alive.numel() == 0:
                        break
                    carry, loss, metrics, sub_preds, _ = train_state.model(
                        carry=carry,
                        batch=sub_batch,
                        return_keys=return_keys,
                        effective_halt_max_steps=eff_limit_t,
                    )
                    inference_steps += 1

                    if not final_preds:
                        for k, v in sub_preds.items():
                            final_preds[k] = torch.empty((B, *v.shape[1:]), dtype=v.dtype, device=v.device)

                    q_vals = sub_preds.get("q_halt_logits")
                    if q_vals is None:
                        # If no q values, just write outputs for all remaining and stop
                        for k, v in sub_preds.items():
                            final_preds[k][global_alive] = v
                        global_alive = global_alive[:0]
                        break

                    finish_mask = (q_vals > 0)
                    if t == max_steps - 1:
                        finish_mask = torch.ones_like(finish_mask, dtype=torch.bool)

                    if finish_mask.any():
                        finished_global = global_alive[finish_mask]
                        for k, v in sub_preds.items():
                            final_preds[k][finished_global] = v[finish_mask]
                        step_counts[finished_global] = t + 1

                    keep_mask = ~finish_mask
                    if keep_mask.any():
                        # Narrow carry and sub_batch by local keep_mask
                        idx_keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
                        global_alive = global_alive[keep_mask]
                        # Update carry tensors in place to only continuing samples
                        carry.inner_carry.z_H = carry.inner_carry.z_H.index_select(0, idx_keep)
                        carry.inner_carry.z_L = carry.inner_carry.z_L.index_select(0, idx_keep)
                        carry.steps = carry.steps.index_select(0, idx_keep)
                        carry.halted = carry.halted.index_select(0, idx_keep)
                        carry.prev_q = carry.prev_q.index_select(0, idx_keep)
                        carry.current_data = {k: v.index_select(0, idx_keep) for k, v in carry.current_data.items()}
                        # Update sub_batch from original full batch using new global indices to avoid drift
                        sub_batch = {k: batch[k].index_select(0, global_alive) for k, v in sub_batch.items()}
                    else:
                        # Everyone finished
                        global_alive = global_alive[:0]
                        break

                preds = final_preds
                # Build minimal metrics from finalized predictions (keeps evaluator flow intact)
                labels = batch.get("labels")
                if labels is not None and "preds" in preds:
                    mask_tok = (labels != -100)
                    valid_counts = mask_tok.sum(-1)
                    count = (valid_counts > 0).sum()
                    correct_tok = (preds["preds"] == labels) & mask_tok
                    frac_correct = torch.where(valid_counts > 0, correct_tok.sum(-1) / valid_counts.clamp_min(1), torch.zeros_like(valid_counts, dtype=torch.float32))
                    accuracy_sum = frac_correct.sum()
                    exact = (correct_tok.sum(-1) == valid_counts) & (valid_counts > 0)
                    exact_sum = exact.sum()
                    q_logits = preds.get("q_halt_logits")
                    q_acc = ((q_logits >= 0) == exact).sum() if q_logits is not None else torch.tensor(0, device="cuda")
                    steps_sum = step_counts.to(torch.float32).sum()
                    metrics = {
                        "count": count.to(torch.float32),
                        "accuracy": accuracy_sum.to(torch.float32),
                        "exact_accuracy": exact_sum.to(torch.float32),
                        "q_halt_accuracy": q_acc.to(torch.float32),
                        "steps": steps_sum,
                        "lm_loss": torch.tensor(0.0, device="cuda"),
                        "q_halt_loss": torch.tensor(0.0, device="cuda"),
                    }
                else:
                    metrics = {
                        "count": torch.tensor(float(B), device="cuda"),
                        "accuracy": torch.tensor(0.0, device="cuda"),
                        "exact_accuracy": torch.tensor(0.0, device="cuda"),
                        "q_halt_accuracy": torch.tensor(0.0, device="cuda"),
                        "steps": step_counts.to(torch.float32).sum(),
                        "lm_loss": torch.tensor(0.0, device="cuda"),
                        "q_halt_loss": torch.tensor(0.0, device="cuda"),
                    }
                all_finish = True
                print(f" Completed partial-finish inference in {inference_steps} steps, metrics: {metrics}")
            else:
                # Default mode: run full-batch until all_finish is True
                while True:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry,
                        batch=batch,
                        return_keys=return_keys,
                        effective_halt_max_steps=eff_limit_t,
                    )
                    inference_steps += 1

                    if all_finish:
                        break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # --- Evaluation‑only mode -------------------------------------------------
    if config.eval_only:
        if RANK == 0:
            print("EVALUATION ONLY - no training will be performed")
        # If EMA is enabled, switch to the EMA weights before evaluating
        if config.ema and ema_helper is not None:
            print("SWITCH TO EMA")
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            train_state_eval = train_state

        train_state_eval.model.eval()
        metrics = evaluate(
            config,
            train_state_eval,
            eval_loader,
            eval_metadata,
            evaluators,
            rank=RANK,
            world_size=WORLD_SIZE,
            cpu_group=CPU_PROCESS_GROUP,
        )

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state_eval.step)

        # Optionally save checkpoint of evaluated model
        if RANK == 0:
            save_train_state(config, train_state_eval)

        # Clean up and exit early
        if dist.is_initialized():
            dist.destroy_process_group()
        wandb.finish()
        return
    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()

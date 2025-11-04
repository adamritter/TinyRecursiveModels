#!/usr/bin/env python3
"""
Evaluate a trained TRM checkpoint on a SAT dataset.

This script runs inference-only: it loads the specified checkpoint, iterates over
the evaluation split, accumulates the loss-head metrics, and optionally feeds the
batch outputs to any configured evaluators.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import hydra
import pydantic
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class EvalConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []

    global_batch_size: int
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    eval_save_outputs: List[str] = []
    eval_split: str = "test"

    seed: int = 0
    project_name: Optional[str] = None
    run_name: Optional[str] = None


def create_dataloader(
    config: EvalConfig,
    split: str,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, PuzzleDatasetMetadata]:
    dataset_paths = config.data_paths_test if split != "train" and config.data_paths_test else config.data_paths
    dataset_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=dataset_paths,
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=rank,
        num_replicas=world_size,
    )
    dataset = PuzzleDataset(dataset_config, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    if isinstance(state_dict, dict) and all(key.startswith("_orig_mod.") for key in state_dict.keys()):
        cleaned = {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        return

    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")


def create_model(
    config: EvalConfig,
    metadata: PuzzleDatasetMetadata,
    device: torch.device,
    rank: int,
    world_size: int,
) -> nn.Module:
    model_kwargs = dict(
        **config.arch.__pydantic_extra__,  # type: ignore[arg-type]
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        base_model = model_cls(model_kwargs)
        model = loss_cls(base_model, **config.arch.loss.__pydantic_extra__)  # type: ignore[arg-type]
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore[attr-defined]
        model.to(device)

    if config.load_checkpoint is None:
        raise ValueError("load_checkpoint must be specified for evaluation.")

    if rank == 0:
        print(f"Loading checkpoint {config.load_checkpoint}")
        load_checkpoint(model, config.load_checkpoint, device)

    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    return model


def create_evaluators(config: EvalConfig, metadata: PuzzleDatasetMetadata) -> List[Any]:
    evaluators: List[Any] = []
    data_paths = config.data_paths_test if config.data_paths_test else config.data_paths
    for evaluator_cfg in config.evaluators:
        evaluator_cls = load_model_class(evaluator_cfg.name, "evaluators.")
        for data_path in data_paths:
            evaluators.append(
                evaluator_cls(
                    data_path=data_path,
                    eval_metadata=metadata,
                    **evaluator_cfg.__pydantic_extra__,
                )
            )
    return evaluators


def evaluate(
    config: EvalConfig,
    model: nn.Module,
    dataloader: DataLoader,
    metadata: PuzzleDatasetMetadata,
    evaluators: Sequence[Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    return_keys = set(config.eval_save_outputs)
    for evaluator in evaluators:
        evaluator.begin_eval()
        return_keys.update(evaluator.required_outputs)

    save_preds: Dict[str, List[torch.Tensor]] = {k: [] for k in config.eval_save_outputs}
    metric_keys: Optional[List[str]] = None
    metric_totals: Dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        for set_name, batch, _global_batch_size in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)  # type: ignore[attr-defined]

            while True:
                carry, _loss, metrics, preds, all_finish = model(  # type: ignore[operator]
                    carry=carry,
                    batch=batch,
                    return_keys=return_keys,
                )

                if metric_keys is None:
                    metric_keys = list(sorted(metrics.keys()))

                metric_vector = torch.stack([metrics[k] for k in metric_keys])
                if set_name not in metric_totals:
                    metric_totals[set_name] = torch.zeros_like(metric_vector)
                metric_totals[set_name] += metric_vector

                for key, values in preds.items():
                    if key in save_preds:
                        save_preds[key].append(values.detach().cpu())

                for evaluator in evaluators:
                    evaluator.update_batch(batch, preds)

                if all_finish:
                    break

    if metric_keys is None:
        metric_keys = []

    set_names = list(metric_totals.keys())
    if world_size > 1 and set_names:
        stacked = torch.stack([metric_totals[name] for name in set_names], dim=0)
        dist.reduce(stacked, dst=0)
        if rank == 0:
            for idx, name in enumerate(set_names):
                metric_totals[name] = stacked[idx]

    evaluator_results: Dict[str, float] = {}
    for evaluator in evaluators:
        result = evaluator.result(
            save_path=config.checkpoint_path if rank == 0 else None,
            rank=rank,
            world_size=world_size,
            group=None,
        )
        if rank == 0 and result:
            evaluator_results.update(result)

    results: Dict[str, Dict[str, float]] = {}
    if rank == 0:
        for set_name in set_names:
            totals = metric_totals[set_name].cpu()
            metrics = dict(zip(metric_keys, totals))
            count = metrics.pop("count", torch.tensor(0.0)).item()
            if count <= 0:
                results[set_name] = {k: float(v.item()) for k, v in metrics.items()}
                results[set_name]["count"] = 0.0
            else:
                averaged = {k: float(v.item() / count) for k, v in metrics.items()}
                averaged["count"] = float(count)
                results[set_name] = averaged

        if config.eval_save_outputs and any(save_preds.values()) and config.checkpoint_path is not None:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            merged = {k: torch.cat(v, dim=0) for k, v in save_preds.items() if v}
            output_path = os.path.join(config.checkpoint_path, "eval_outputs.pt")
            torch.save(merged, output_path)
            print(f"Saved evaluation outputs to {output_path}")

        if evaluator_results:
            results.update({"evaluators": evaluator_results})

    return results


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> EvalConfig:
    container = [None]
    if rank == 0:
        container[0] = EvalConfig(**hydra_config)  # type: ignore[arg-type]
    if world_size > 1:
        dist.broadcast_object_list(container, src=0)
    return container[0]  # type: ignore[return-value]


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig) -> None:
    rank = 0
    world_size = 1
    local_rank_env = os.environ.get("LOCAL_RANK")

    if local_rank_env is not None:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(local_rank_env))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and local_rank_env is not None:
        device = torch.device("cuda", int(local_rank_env))

    config = load_synced_config(hydra_config, rank, world_size)
    torch.manual_seed(config.seed + rank)

    dataloader, metadata = create_dataloader(config, config.eval_split, rank, world_size)
    model = create_model(config, metadata, device, rank, world_size)
    evaluators = create_evaluators(config, metadata)
    results = evaluate(config, model, dataloader, metadata, evaluators, device, rank, world_size)

    if rank == 0:
        if not results:
            print("No metrics were produced during evaluation.")
        else:
            for set_name, metrics in results.items():
                print(f"[{set_name}]")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.6f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

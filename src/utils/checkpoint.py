"""Checkpoint management for training state persistence."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from typing import Optional

from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder
from src.model.vocab import Vocabulary


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_loss: float


class CheckpointManager:
    """管理训练检查点的保存和加载。"""

    @staticmethod
    def save_checkpoint(
        checkpoint_dir: Path | str,
        model: EditFlowsTransformer,
        encoder: SetEncoder,
        optimizer: torch.optim.Optimizer,
        vocab: Vocabulary,
        training_state: TrainingState,
    ):
        """保存完整训练状态。

        Args:
            checkpoint_dir: 检查点保存目录
            model: EditFlowsTransformer 模型
            encoder: SetEncoder 编码器
            optimizer: 优化器
            vocab: 词汇表
            training_state: 训练状态
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型配置
        model_config = {
            "vocab_size": model.vocab_size,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "max_seq_len": model.max_seq_len,
            "bos_token_id": model.bos_token_id,
            "pad_token_id": model.pad_token_id,
            "num_cond_features": model.num_cond_features,
        }
        with open(checkpoint_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        # 保存编码器配置
        encoder_config = {
            "dim_input": encoder.linearl.in_features,
            "dim_hidden": encoder.selfatt1.mab1.dim_V,
            "num_heads": encoder.selfatt1.mab1.num_heads,
            "num_inds": encoder.selfatt1.I.shape[1],
            "ln": hasattr(encoder.selfatt1.mab1, "ln0"),
            "n_l_enc": len(encoder.selfatt),
            "num_features": encoder.outatt.S.shape[1],
            "linear": encoder.linear,
        }
        with open(checkpoint_dir / "encoder_config.json", "w") as f:
            json.dump(encoder_config, f, indent=2)

        # 保存词汇表配置
        vocab.save(checkpoint_dir / "vocab_config.json")

        # 保存权重
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        torch.save(encoder.state_dict(), checkpoint_dir / "encoder.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # 保存训练状态
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(asdict(training_state), f, indent=2)

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path | str,
        model: EditFlowsTransformer,
        encoder: SetEncoder,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
    ) -> TrainingState:
        """恢复训练状态。

        Args:
            checkpoint_path: 检查点目录路径
            model: EditFlowsTransformer 模型实例
            encoder: SetEncoder 编码器实例
            optimizer: 优化器实例
            device: 设备（可选，默认为 'cpu'）

        Returns:
            TrainingState: 训练状态
        """
        checkpoint_path = Path(checkpoint_path)

        map_location = device if device else 'cpu'
        # 加载权重（strict=False 允许部分加载，用于模型架构迁移场景）
        model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location=map_location), strict=False)
        encoder.load_state_dict(torch.load(checkpoint_path / "encoder.pt", map_location=map_location))

        # 加载 optimizer（如果存在）
        if (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location=map_location))

        # 加载训练状态（如果存在）
        if (checkpoint_path / "training_state.json").exists():
            with open(checkpoint_path / "training_state.json", "r") as f:
                state_dict = json.load(f)
            return TrainingState(**state_dict)

        # 如果只有模型权重，返回初始状态
        return TrainingState(epoch=0, global_step=0, best_loss=float('inf'))

    @staticmethod
    def save_model_only(
        save_dir: Path | str,
        model: EditFlowsTransformer,
        encoder: SetEncoder,
        vocab: Vocabulary,
    ):
        """仅保存模型（用于推理）。

        Args:
            save_dir: 保存目录
            model: EditFlowsTransformer 模型
            encoder: SetEncoder 编码器
            vocab: 词汇表
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型配置
        model_config = {
            "vocab_size": model.vocab_size,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "max_seq_len": model.max_seq_len,
            "bos_token_id": model.bos_token_id,
            "pad_token_id": model.pad_token_id,
            "num_cond_features": model.num_cond_features,
        }
        with open(save_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        # 保存编码器配置
        encoder_config = {
            "dim_input": encoder.linearl.in_features,
            "dim_hidden": encoder.selfatt1.mab1.dim_V,
            "num_heads": encoder.selfatt1.mab1.num_heads,
            "num_inds": encoder.selfatt1.I.shape[1],
            "ln": hasattr(encoder.selfatt1.mab1, "ln0"),
            "n_l_enc": len(encoder.selfatt),
            "num_features": encoder.outatt.S.shape[1],
            "linear": encoder.linear,
        }
        with open(save_dir / "encoder_config.json", "w") as f:
            json.dump(encoder_config, f, indent=2)

        # 保存词汇表配置
        vocab.save(save_dir / "vocab_config.json")

        # 保存权重
        torch.save(model.state_dict(), save_dir / "model.pt")
        torch.save(encoder.state_dict(), save_dir / "encoder.pt")

    @staticmethod
    def load_model_only(
        load_path: Path | str,
        device: torch.device,
    ) -> tuple[EditFlowsTransformer, SetEncoder, Vocabulary]:
        """仅加载模型（用于推理）。

        Args:
            load_path: 模型目录路径
            device: 设备

        Returns:
            (model, encoder, vocab): 模型、编码器、词汇表
        """
        load_path = Path(load_path)

        # 加载配置
        with open(load_path / "model_config.json", "r") as f:
            model_config = json.load(f)

        with open(load_path / "encoder_config.json", "r") as f:
            encoder_config = json.load(f)

        # 加载词汇表
        vocab = Vocabulary.load(load_path / "vocab_config.json")

        # 创建模型实例
        model = EditFlowsTransformer(**model_config).to(device)
        encoder = SetEncoder(**encoder_config).to(device)

        # 加载权重
        model.load_state_dict(torch.load(load_path / "model.pt", map_location=device))
        encoder.load_state_dict(torch.load(load_path / "encoder.pt", map_location=device))

        model.eval()
        encoder.eval()

        return model, encoder, vocab

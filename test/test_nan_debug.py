"""NaN 数值爆炸调试测试。

逐步检查训练过程中各变量的数值状态，定位 NaN 产生的具体位置。
"""

import sys
sys.path.insert(0, '/home/xyh/Symbolic_Regression/Diffusion-Based/SGB-EF_Rebase')

import torch
import torch.nn.functional as F
from src.data_loader.data_loader import SRDataLoader
from src.model.data_embedding import SetEncoder, prepare_encoder_input
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.vocab import Vocabulary
from src.core.flow_helper import (
    x2prob,
    sample_cond_pt,
    CubicScheduler,
    make_ut_mask_from_z,
    rm_gap_tokens,
    fill_gap_tokens_with_repeats,
)


def print_tensor_stats(name, tensor, indent="  ", verbose=True):
    """打印张量统计信息"""
    if tensor is None:
        if verbose:
            print(f"{indent}{name}: None")
        return

    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    finite_count = (~torch.isnan(tensor) & ~torch.isinf(tensor)).sum().item()
    total = tensor.numel()

    if nan_count > 0 or inf_count > 0:
        stats = {
            "shape": list(tensor.shape),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        print(f"{indent}{name}: ⚠️  NaN={nan_count}, Inf={inf_count}, shape={stats['shape']}")
    elif verbose:
        stats = {
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "shape": list(tensor.shape),
        }
        print(f"{indent}{name}: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, shape={stats['shape']}")


def print_model_params(model, name, indent="  "):
    """打印模型参数统计"""
    print(f"{indent}{name} 参数:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            nan_count = torch.isnan(p).sum().item()
            inf_count = torch.isinf(p).sum().item()
            grad = p.grad
            grad_stats = ""
            if grad is not None:
                g_nan = torch.isnan(grad).sum().item()
                g_inf = torch.isinf(grad).sum().item()
                if g_nan > 0 or g_inf > 0:
                    grad_stats = f", grad: NaN={g_nan}, Inf={g_inf}"

            if nan_count > 0 or inf_count > 0:
                print(f"{indent}  ⚠️  {n}: NaN={nan_count}, Inf={inf_count}, shape={list(p.shape)}{grad_stats}")


def main():
    print("=" * 60)
    print("NaN 数值爆炸调试测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 1. 加载数据
    print("\n" + "=" * 40)
    print("1. 加载数据集")
    print("=" * 40)
    data_loader = SRDataLoader(
        npz_path='data/data_100_3v.npz',
        batch_size=2,
        shuffle=False,  # 使用固定顺序以便重现
        device=device,
    )
    print(f"数据集大小: {len(data_loader.dataset)}")
    print(f"批次数: {len(data_loader)}")

    # 获取第一个 batch 以确定 input_dim
    first_batch = next(iter(data_loader))
    x_0, x_1, z_0, z_1, t, x_values, y_target = first_batch
    input_dim = x_values.shape[-1]
    print(f"Input dimension: {input_dim}")

    # 重新创建 loader 以从头开始
    data_loader = SRDataLoader(
        npz_path='data/data_100_3v.npz',
        batch_size=2,
        shuffle=False,
        device=device,
    )

    # 2. 创建 Vocab
    print("\n" + "=" * 40)
    print("2. 创建 Vocabulary")
    print("=" * 40)
    vocab = Vocabulary(num_variables=input_dim)
    print(f"Vocab size: {vocab.vocab_size}")

    # 3. 创建模型
    print("\n" + "=" * 40)
    print("3. 创建模型")
    print("=" * 40)

    data_encoder = SetEncoder(
        dim_input=input_dim + 1,
        dim_hidden=128,
        num_heads=4,
        num_inds=32,
        ln=True,
        n_l_enc=2,
        num_features=1,
        linear=True,
    ).to(device)

    model = EditFlowsTransformer(
        vocab_size=vocab.vocab_size,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        bos_token_id=vocab.token_to_id('<s>'),
        pad_token_id=vocab.pad_token,
        num_cond_features=1,
    ).to(device)

    # 4. 创建 Optimizer 和 Scheduler
    print("\n" + "=" * 40)
    print("4. 创建 Optimizer 和 Scheduler")
    print("=" * 40)
    all_params = list(model.parameters()) + list(data_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)
    flow_scheduler = CubicScheduler(a=1.0, b=1.0)

    # 5. 逐步执行训练循环
    print("\n" + "=" * 40)
    print("5. 逐步执行训练循环")
    print("=" * 40)

    max_batches = 100  # 运行更多 batch 以捕获问题
    verbose_batches = {0, 1, 2, 3, 4, 20, 21}  # 前 5 个 batch 和 batch 21 (出错前) 详细输出
    found_nan = False

    for step, batch in enumerate(data_loader):
        if step >= max_batches:
            break

        verbose = step in verbose_batches or step % 10 == 0
        if verbose:
            print(f"\n--- Batch {step + 1}/{max_batches} ---")

        x_0, x_1, z_0, z_1, t, x_values, y_target = batch

        # 5.1 检查输入数据
        if verbose:
            print("  [1] 输入数据:")
            print_tensor_stats("x_values", x_values, "    ", verbose)
            print_tensor_stats("y_target", y_target, "    ", verbose)

        # 5.2 编码器前向
        if verbose:
            print("  [2] 编码器前向:")

        encoder_input = prepare_encoder_input(x_values, y_target)
        condition = data_encoder(encoder_input)

        if verbose:
            print_tensor_stats("condition", condition, "    ", verbose)

        if torch.isnan(condition).any():
            print(f"    ⚠️  [Batch {step}] 检测到 NaN 在编码器输出!")
            print_tensor_stats("condition", condition, "    ", True)
            print_model_params(data_encoder, "SetEncoder (after forward)", "    ")
            found_nan = True
            break

        # 5.3 流采样
        if verbose:
            print("  [3] 流采样:")

        p0 = x2prob(z_0, vocab.vocab_size)
        p1 = x2prob(z_1, vocab.vocab_size)
        z_t = sample_cond_pt(p0, p1, t, flow_scheduler)

        if verbose:
            print_tensor_stats("z_t", z_t.float(), "    ", verbose)

        # 5.4 移除 gap token
        if verbose:
            print("  [4] 移除 gap token:")

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, vocab.pad_token, vocab.gap_token)

        if verbose:
            print(f"    x_t: shape={x_t.shape}, x_pad_mask: {x_pad_mask.sum().item()}, z_gap_mask: {z_gap_mask.sum().item()}")

        # 5.5 创建 u_t mask
        uz_mask = make_ut_mask_from_z(
            z_t, z_1,
            vocab_size=vocab.vocab_size,
            pad_token=vocab.pad_token,
            gap_token=vocab.gap_token,
        )

        # 5.6 模型前向
        if verbose:
            print("  [6] 模型前向:")

        try:
            u_t, ins_probs, sub_probs = model.forward(
                tokens=x_t,
                time_step=t,
                padding_mask=x_pad_mask,
                condition=condition,
            )
            if verbose:
                print_tensor_stats("u_t", u_t, "    ", verbose)
                print_tensor_stats("ins_probs", ins_probs, "    ", verbose)
                print_tensor_stats("sub_probs", sub_probs, "    ", verbose)
        except ValueError as e:
            print(f"    ⚠️  [Batch {step}] 模型前向报错: {e}")

            # 调试：检查模型内部状态
            print("    检查嵌入层:")
            token_emb = model.token_embedding(x_t)
            print_tensor_stats("token_emb", token_emb, "    ", True)
            time_emb = model.time_embedding(t)
            print_tensor_stats("time_emb", time_emb, "    ", True)
            cond_proj = model.cond_proj(condition)
            print_tensor_stats("cond_proj", cond_proj, "    ", True)

            print_model_params(model, "EditFlowsTransformer (at error)", "    ")
            found_nan = True
            break

        # 5.7 计算损失
        if verbose:
            print("  [7] 计算损失:")

        lambda_ins = u_t[:, :, 0]
        lambda_sub = u_t[:, :, 1]
        lambda_del = u_t[:, :, 2]

        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs
        u_tia_del = lambda_del.unsqueeze(-1)

        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)

        # 调试：检查除零问题
        kappa_t = flow_scheduler(t)
        one_minus_kappa = 1 - kappa_t
        derivative_t = flow_scheduler.derivative(t)

        if verbose or step == 21:
            print_tensor_stats("kappa(t)", kappa_t, "    ", True)
            print_tensor_stats("1-kappa(t)", one_minus_kappa, "    ", True)
            print_tensor_stats("derivative(t)", derivative_t, "    ", True)

        if (one_minus_kappa < 1e-6).any():
            print(f"    ⚠️  [Batch {step}] 1 - kappa(t) 接近零: min_kappa={kappa_t.min():.6f}, min_one_minus={one_minus_kappa.min():.6e}")

        sched_coeff = (derivative_t / one_minus_kappa).to(device)
        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
        u_tot = u_t.sum(dim=(1, 2))
        masked_log = (log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = (u_tot - masked_log).mean()

        if verbose or step == 21:
            print_tensor_stats("sched_coeff", sched_coeff, "    ", True)
            print_tensor_stats("u_tot", u_tot, "    ", True)
            print_tensor_stats("masked_log", masked_log, "    ", True)
            print_tensor_stats("loss", loss.unsqueeze(0), "    ", True)

        # 检查损失是否异常大
        if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
            print(f"    ⚠️  [Batch {step}] 损失异常: {loss.item():.6f}")
            print_tensor_stats("sched_coeff", sched_coeff, "    ", True)
            print_tensor_stats("u_tot", u_tot, "    ", True)
            print_tensor_stats("masked_log", masked_log, "    ", True)
            print_tensor_stats("uz_cat", uz_cat, "    ", True)
            found_nan = True
            break

        # 5.8 反向传播
        if verbose or step >= 20:
            print("  [8] 反向传播:")

        optimizer.zero_grad()

        # 计算损失前的梯度范数
        if step >= 20:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"    backward 前 grad_norm: {total_norm:.6f}")

        loss.backward()

        # 计算损失后的梯度范数
        if step >= 20:
            total_norm = 0
            nan_count = 0
            inf_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        nan_count += 1
                    if torch.isinf(p.grad).any():
                        inf_count += 1
                    param_norm = p.grad.data.norm(2)
                    if not torch.isnan(param_norm) and not torch.isinf(param_norm):
                        total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"    backward 后 grad_norm: {total_norm:.6f}, nan_grads: {nan_count}, inf_grads: {inf_count}")

        # 检查梯度
        has_nan_grad = False
        for n, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"    ⚠️  [Batch {step}] 梯度异常: {n}")
                    has_nan_grad = True

        for n, p in data_encoder.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"    ⚠️  [Batch {step}] 梯度异常 (encoder): {n}")
                    has_nan_grad = True

        if has_nan_grad:
            print("    ⚠️  检测到 NaN/Inf 梯度!")
            found_nan = True
            break

        # 5.9 优化器步进
        if verbose or step >= 20:
            print("  [9] 优化器步进:")

        optimizer.step()

        # 检查参数更新后状态 - 在 batch 22 时详细检查
        param_ok = True
        param_nan_count = 0
        param_inf_count = 0
        param_max_abs = 0
        param_max_name = ""

        for n, p in model.named_parameters():
            nan_count = torch.isnan(p).sum().item()
            inf_count = torch.isinf(p).sum().item()
            max_abs = p.abs().max().item()
            if max_abs > param_max_abs:
                param_max_abs = max_abs
                param_max_name = f"model.{n}"

            if nan_count > 0:
                param_nan_count += 1
                print(f"    ⚠️  [Batch {step}] 参数 NaN: {n}, count={nan_count}")
                param_ok = False
            if inf_count > 0:
                param_inf_count += 1
                print(f"    ⚠️  [Batch {step}] 参数 Inf: {n}, count={inf_count}")
                param_ok = False

        for n, p in data_encoder.named_parameters():
            nan_count = torch.isnan(p).sum().item()
            inf_count = torch.isinf(p).sum().item()
            max_abs = p.abs().max().item()
            if max_abs > param_max_abs:
                param_max_abs = max_abs
                param_max_name = f"encoder.{n}"

            if nan_count > 0:
                param_nan_count += 1
                print(f"    ⚠️  [Batch {step}] 参数 NaN (encoder): {n}, count={nan_count}")
                param_ok = False
            if inf_count > 0:
                param_inf_count += 1
                print(f"    ⚠️  [Batch {step}] 参数 Inf (encoder): {n}, count={inf_count}")
                param_ok = False

        if step >= 20:
            print(f"    参数状态: ok={param_ok}, nan={param_nan_count}, inf={param_inf_count}, max_abs={param_max_abs:.2e} ({param_max_name})")

        # 检查参数是否异常大（接近 Inf）
        if param_max_abs > 1e5:
            print(f"    ⚠️  [Batch {step}] 参数异常大! max_abs={param_max_abs:.2e}")
            found_nan = True
            break

        if not param_ok:
            print("    ⚠️  检测到 NaN/Inf 参数!")
            print_model_params(model, "EditFlowsTransformer (after step)", "    ")
            print_model_params(data_encoder, "SetEncoder (after step)", "    ")
            found_nan = True
            break

        if verbose:
            print(f"    损失: {loss.item():.6f}")
        elif step % 5 == 0:
            # 每 5 个 batch 打印一次损失
            print(f"    [Batch {step + 1}] loss: {loss.item():.6f}")

    print("\n" + "=" * 60)
    if found_nan:
        print("⚠️  检测到数值异常，已定位到问题位置")
    else:
        print(f"✓ 完成 {max_batches} 个 batch，未检测到数值异常")
    print("=" * 60)


if __name__ == '__main__':
    main()

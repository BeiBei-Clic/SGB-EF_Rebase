# SGB-EF 符号回归项目

基于 Edit Flows 的符号回归实现。

## 快速开始

### 数据准备

```bash
# 将文本数据转换为 npz 格式
uv run python src/utils/txt2Npz.py data/flow_samples_10000_3dim_100pts_6depth_12len.txt data/flow_samples_10000_3dim_100pts_6depth_12len.npz

# 生成训练数据
uv run python -m data_generator.data_generator --num-samples 1

# 多核（使用全部 CPU）
python -m data_generator.data_generator --num-samples 10000 --num-workers -1

# 单核（原方式）
python -m data_generator.data_generator --num-samples 10000 --num-workers 1
```

### 模型训练

```bash
uv run python train.py --epochs 1000 --lr 1e-4 --checkpoint-every 500 --data-path data/data_1_3v.npz

uv run accelerate launch --num_processes 1 train.py --data-path data/data_10000_3v.npz --epochs 100 --batch-size 32

CUDA_VISIBLE_DEVICES=0
```

### 推理与采样

```bash
# 单样本推理
uv run python inference.py --sample-idx 0 --n-steps 500 --data-path data/data_1_3v.npz
```

### 数据查看

```bash
# 查看数据集内容
uv run python -m data_generator.view_data --path data/data_1_3v.npz --num-samples 1
```

### 测试与调试

```bash
# 运行 NaN 调试工具
uv run python test/test_nan_debug.py
```

### 监控工具

```bash
# GPU 监控
watch -n 1 nvidia-smi
```

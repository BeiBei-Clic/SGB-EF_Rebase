uv run python src/utils/txt2Npz.py data/flow_samples_10000_3dim_100pts_6depth_12len.txt data/flow_samples_10000_3dim_100pts_6depth_12len.npz

uv run python train.py --epochs 100000000 --lr 1e-4 --checkpoint-every 50000000 --data-path data/data_100_3v.npz

uv run python inference.py --sample-idx 0 --n-steps 500

watch -n 1 nvidia-smi

uv run python -m data_generator.view_data --path data/train_data.npz --num-samples 100

uv run python -m data_generator.data_generator --num-samples 100

uv run python test/test_nan_debug.py
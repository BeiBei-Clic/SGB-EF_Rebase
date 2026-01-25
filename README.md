uv run python src/utils/txt2Npz.py data/flow_samples_10000_3dim_100pts_6depth_12len.txt data/flow_samples_10000_3dim_100pts_6depth_12len.npz

uv run python train.py --epochs 100000000 --lr 1e-4 --checkpoint-every 50000000

 uv run python inference.py --sample-idx 0 --n-steps 500

   watch -n 1 nvidia-smi
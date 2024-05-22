# Multi-node DDP
python main.py train {CONFIG} --num-workers 4 \
    --gpus 1 --num_nodes 4 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

# Single-node DDP
ts python main.py train configs/mks/mk015.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

# Single GPU
python main.py train {CONFIG} --num-workers 4 \
    --gpus 1 --num_nodes 1 --precision 16 \
    --benchmark --kfold 0

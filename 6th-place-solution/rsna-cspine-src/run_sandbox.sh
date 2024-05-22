ts python main.py train configs/mk3d/mk3d010.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 5

ts python main.py train configs/mk3d/mk3d010.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1 --check_val_every_n_epoch 5

ts python main.py train configs/mk3d/mk3d010.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2 --check_val_every_n_epoch 5

ts python main.py train configs/mk3d/mk3d010.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3 --check_val_every_n_epoch 5

ts python main.py train configs/mk3d/mk3d010.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4 --check_val_every_n_epoch 5
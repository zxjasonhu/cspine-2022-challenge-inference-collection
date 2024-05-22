# Multi-node DDP
python main.py predict {CONFIG} --num-workers 4 \
    --gpus 1 --num_nodes 4 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

# Single-node DDP
ts python main.py predict configs/pre/pre000.yaml --num-workers 4 \
    --gpus 1 --num_nodes 1 \
    --inference-checkpoint ../experiments/pre000/sbn/fold0_seed890/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles positive_fracture_candidate_slices.txt \
    --inference-save ../predictions/pre000/fold0_seed890.pkl \
    --benchmark \
    --inference-tta 5crop

# Single GPU
python main.py predict {CONFIG} --num-workers 4 \
    --gpus 1 --num_nodes 1 --precision 16 \
    --benchmark --kfold 0



ts python main.py predict configs/pre/pre004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre004/sbn/fold0_seed870/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles sample.txt \
    --inference-save ../predictions/pre004/sample.pkl \
    --benchmark

ts python main.py predict configs/pre/pre004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre004/sbn/fold0_seed880/checkpoints/last.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre004/fold0_seed880.pkl \
    --inference-tta-5crop --inference-act-fn sigmoid \
    --benchmark

ts python main.py predict configs/pre/pre004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre004/sbn/fold0_seed890/checkpoints/last.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre004/fold0_seed890.pkl \
    --inference-tta-5crop --inference-act-fn sigmoid \
    --benchmark

ts python main.py predict configs/pre/pre000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre000/sbn/fold1/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre000/fold1.pkl \
    --benchmark

ts python main.py predict configs/pre/pre000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre000/sbn/fold2/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre000/fold2.pkl \
    --benchmark

ts python main.py predict configs/pre/pre000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre000/sbn/fold3/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre000/fold3.pkl \
    --benchmark

ts python main.py predict configs/pre/pre000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp \
    --inference-checkpoint ../experiments/pre000/sbn/fold4/checkpoints/best.ckpt \
    --inference-data-dir ../data/pngs/ \
    --inference-imgfiles unlabeled_positive_slices_2dc.txt \
    --inference-save ../predictions/pre000/fold4.pkl \
    --benchmark
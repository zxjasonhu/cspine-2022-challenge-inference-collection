from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from skp import builder
from skp.controls.datamaker import get_train_val_datasets


cfg = OmegaConf.load("configs/mk3d/mk3d000.yaml")
train_dataset, val_dataset = get_train_val_datasets(cfg)

loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)

for X, y in loader:
    break


model = builder.build_model(cfg)
out = model(X)
loss_fn = builder.build_loss(cfg)
loss = loss_fn(out, y)

test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
for X, y in test_loader:
    break


from monai.inferers import sliding_window_inference
import torch
with torch.no_grad():
    model.cuda()
    out = sliding_window_inference(X.cuda(), [128, 128, 128], sw_batch_size=1, predictor=model.eval())
import torch

from models.engine import Net2D
from models.sequence import Transformer


# Test various Net2D configurations
model_params = {
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "num_classes": 3,
    "dropout": 0.2,
    "pool": "avgmax"
}
model = Net2D(**model_params)
x = torch.randn((2,3,128,128))
features = model.extract_features(x)
out = model(x) 

model_params.update({"pool": "gem"})
model = Net2D(**model_params)
features = model.extract_features(x)
out = model(x) 

model_params.update({"pool": "catavgmax"})
model = Net2D(**model_params)
features = model.extract_features(x)
out = model(x) 

model_params.update({"feature_reduction": 128})
model = Net2D(**model_params)
features = model.extract_features(x)
out = model(x) 

model_params.update({"multisample_dropout": True})
model = Net2D(**model_params)
features = model.extract_features(x)
out = model(x) 

# Test Transformer
model = Transformer(num_classes=3)
x, mask = torch.randn((2,64,512)), torch.ones((2,64))
out = model((x, mask))

model = Transformer(num_classes=3, predict_sequence=False, pool="gem")
out = model((x, mask))

#####
import torch

from models.engine import Net3D


model_params = {
    "backbone": "x3d_s",
    "pretrained": True,
    "num_classes": 3,
    "dropout": 0.2,
    "pool": "avgmax"
}

model = Net3D(**model_params)
x = torch.randn((2,3,64,64,64))
features = model.extract_features(x)
out = model(x) 

#####

import torch

from models.segmentation.encoders.create import create_encoder
from models.segmentation.decoders.deeplabv3.model import DeepLabV3Plus
from models.segmentation.decoders.fpn.model import FPN
from models.segmentation.decoders.unet.model import Unet


x = torch.randn((2,3,128,128))

# Test Swin+DeepLabV3Plus
model = DeepLabV3Plus(encoder_name="swin", 
                      encoder_params={"pretrained": False, "output_stride": 32})
out = model(x)

# Test timm+DeepLabV3Plus
model = DeepLabV3Plus(encoder_name="resnet34d",
                      encoder_params={"pretrained": False, "output_stride": 16})
out = model(x) 

# Test adjusting output stride
encoder = create_encoder("tf_efficientnetv2_b3", encoder_params={"pretrained": False}, encoder_output_stride=16)
out = encoder(x)

model = DeepLabV3Plus(encoder_name="tf_efficientnet_b4",
                      encoder_params={"pretrained": False, "output_stride": 16})
out = model(x) 

model = DeepLabV3Plus(encoder_name="convnext_base_384_in22ft1k", 
                      encoder_params={"pretrained": False, "output_stride": 16})
out = model(x) 

encoder = create_encoder("convnext_base_384_in22ft1k", encoder_params={"pretrained": False}, encoder_output_stride=16)

# Test Unet
model = Unet(encoder_name="tf_efficientnet_b4", encoder_depth=5, decoder_attention_type="scse")
out = model(x) 

# Test FPN
model = FPN(encoder_name="tf_efficientnet_b4", encoder_depth=5)
out = model(x)

# Test Swin-Unet
model = Unet(encoder_name="swin", encoder_depth=4, decoder_channels=[256, 128, 64, 32], decoder_attention_type="scse")
out = model(x) 

# Test Swin-FPN
model = FPN(encoder_name="swin", encoder_depth=4)
out = model(x)

# Test Unet with deep supervision
from losses import SupervisorLoss

y = (torch.randn((2,3,128,128)) > 0.5).float()
model = Unet(encoder_name="tf_efficientnet_b4", classes=3, encoder_params={"pretrained": True, "depth": 5}, decoder_attention_type="scse", deep_supervision=True)
out = model(x)
criterion = SupervisorLoss("DiceLoss", {"mode": "multilabel"})
loss = criterion(out, y)

#####

import torch

from models.backbones import create_x3d, create_backbone
from models.segmentation.encoders.create import create_encoder
from models.segmentation.decoders.deeplabv3_3d.model import DeepLabV3Plus_3D
from models.segmentation.decoders.unet_3d.model import Unet_3D

x = torch.randn((2,3,64,64,64))
backbone, df = create_backbone("x3d_s", pretrained=True, features_only=False)
out = backbone(x)

model = create_x3d("x3d_s", pretrained=True, features_only=True, z_strides=[2,2,2,2,2])
encoder = create_encoder("x3d_s", encoder_params={"pretrained": True, "output_stride": 32, "z_strides": [2,2,2,2,2]})

model = DeepLabV3Plus_3D(encoder_name="x3d_s", deep_supervision=True,
                         encoder_params={"pretrained": True, "output_stride": 16, "z_strides": [2,2,2,2,1]})
out = model(x)

model = Unet_3D(encoder_name="x3d_s", deep_supervision=True,
                encoder_params={"pretrained": True, "depth": 5, "z_strides": [2,2,2,2,2]})
out = model(x)



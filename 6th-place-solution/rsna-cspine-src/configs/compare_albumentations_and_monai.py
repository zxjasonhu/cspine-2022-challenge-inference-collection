import albumentations as A
import monai
import numpy as np


resize_a = A.Compose([A.Resize(192, 192, p=1], p=1, additional_targets={f"image{i}" : "image" for i in range(1, 256)}
resize_m = monai.transforms.Resize([192, 192, 192])


x = np.ones((320, 512, 512))




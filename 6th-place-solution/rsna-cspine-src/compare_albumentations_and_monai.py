import albumentations as A
import numpy as np
import time

from monai import transforms


def time_func(func, args, num_iter=100):
    tic = time.time()
    for i in range(num_iter):
        func(**args)
    toc = time.time() - tic
    return toc / num_iter


resize_a = A.Compose([A.RandomContrast(p=1)], p=1, additional_targets={f"image{i}" : "image" for i in range(1, 256)})
def resize_aa(x):
    x = resize_a(image=x[0], **{f"image{i}": x[i] for i in range(1, len(x))})
    x = np.stack([x["image"]] + [x[f"image{i}"] for i in range(1, len(x))])

resize_m = transforms.RandAdjustContrast(prob
                                         =1)

def resize_mm(x):
    resize_m(x)

x = np.ones((256, 512, 512))

print(time_func(resize_aa, {"x": x}))
print(time_func(resize_mm, {"x": x}))




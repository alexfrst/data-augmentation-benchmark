from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from augmentation import transforms_list

plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('dataset/dataset-train') / 'Alfalfa/Alfalfa.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, image_path=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 1, num_rows * 1))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx].replace('-', '\n'))

    if image_path is not None:
        plt.savefig(image_path, dpi=500)
    else:
        plt.tight_layout()
        plt.show()


print(f"{len(transforms_list)=}")

mid_index = len(transforms_list) // 2

imgs = [
    [augmenter(orig_img) for _ in range(4)]
    for augmenter, name in transforms_list[:mid_index + 2] + transforms_list[-1:]
]

row_title = [name for _, name in transforms_list[:mid_index + 2] + transforms_list[-1:]]
plot(imgs, row_title=row_title, image_path="figures/most_visible_transforms.png")

imgs = [
    [augmenter(orig_img) for _ in range(4)]
    for augmenter, name in transforms_list[:mid_index]
]

row_title = [name for _, name in transforms_list[:mid_index]]
plot(imgs, row_title=row_title, image_path="figures/transforms_1_2.png")

imgs = [
    [augmenter(orig_img) for _ in range(4)]
    for augmenter, name in transforms_list[mid_index:]
]

row_title = [name for _, name in transforms_list[mid_index:]]
plot(imgs, row_title=row_title, image_path="figures/transforms_2_2.png")

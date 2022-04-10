import numpy as np
import plotly.express as px
import torch

from model.evaluate import get_confusion_matrix
from model.model import load_convnext_small
from utils.dataset import load_dataset

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    model = load_convnext_small(62, False)
    model.load_state_dict(torch.load("best_params_convnext_small_convnext_multi_augmentation.pt"))
    model.train(False)
    device = torch.device("cpu")
    model.to(device)
    train_loader, val_loader, test_loader, nb_classes = load_dataset("dataset/dataset-train", batch_size=16)
    confusion_matrix = get_confusion_matrix(model, test_loader, device)
    print(confusion_matrix.shape)
    print(np.trace(confusion_matrix) / confusion_matrix.sum())

    fig = px.imshow(confusion_matrix, color_continuous_scale="Blues", width=400, height=400)
    fig.update_layout(coloraxis_showscale=False, showlegend=False, yaxis_title=None, xaxis_title=None, margin=dict(
        l=0,  # left
        r=0,  # right
        t=0,  # top
        b=0,  # bottom
    ))
    fig.write_image("figures/confusion_matrix.png", scale=4)

import torch.nn as nn
from torchvision import models


def load_inception_for_tl(nb_classes, device):
    inception_v3 = models.inception_v3(pretrained=True)

    # on indique qu'il est inutile de calculer les gradients des paramètres du réseau
    for param in inception_v3.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    # Handle the auxilary net
    num_ftrs = inception_v3.AuxLogits.fc.in_features
    inception_v3.AuxLogits.fc = nn.Linear(num_ftrs, nb_classes)
    # Handle the primary net
    num_ftrs = inception_v3.fc.in_features
    inception_v3.fc = nn.Linear(num_ftrs, nb_classes)

    return inception_v3

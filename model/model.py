import torch.nn as nn
from torchvision import models


def load_inception(nb_classes):
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

def load_mobilenet(nb_classes):
    mobile_net = models.mobilenet_v2(pretrained=True)

    # on indique qu'il est inutile de calculer les gradients des paramètres du réseau
    for param in mobile_net.parameters():
        param.requires_grad = False

    # on remplace la dernière couche fully connected à 1000 sorties (classes d'ImageNet) par une fully connected à 6 sorties (nos classes).
    # par défaut, les gradients des paramètres cette couche seront bien calculés
    mobile_net.classifier[1] = nn.Linear(in_features=mobile_net.classifier[1].in_features, out_features=nb_classes, bias=True)

    return mobile_net

def get_params_tranfer_learning(model):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

if __name__ == '__main__':
    inception_v3 = load_inception(nb_classes=2)
    print(inception_v3._get_name())
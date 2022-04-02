import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model.evaluate import evaluate
from model.model import load_inception, get_params_tranfer_learning
from model.train import train_model
from utils.dataset import load_dataset, load_zipfile
from utils.print_utils import Symbols

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Working on {device} {Symbols.OK}")

    load_zipfile()

    train_loader, val_loader, test_loader, nb_classes = load_dataset("dataset/dataset-train", "dataset/dataset-test")
    inception_v3 = load_inception(nb_classes)
    criterion = nn.CrossEntropyLoss()

    params_to_update = get_params_tranfer_learning(inception_v3)
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5)

    print("Apprentissage en transfer learning")
    inception_v3.to(device)
    inception_v3.train(True)
    torch.manual_seed(42)

    train_model(inception_v3, train_loader, val_loader, scheduler, criterion, evaluate, 65, device=device)
    # train_model(inception_v3, criterion, optimizer, None, num_epochs=65, is_inception=True)
    inception_v3.train(False)
    loss, accuracy = evaluate(inception_v3, test_loader, device, criterion)
    print("Accuracy (test): %.1f%%" % (100 * accuracy))

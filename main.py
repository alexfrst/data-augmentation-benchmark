import torch
import torch.nn as nn
import torch.optim as optim

from model.evaluate import evaluate
from model.model import load_inception_for_tl
from model.train import train_model
from utils.dataset import load_dataset, load_zipfile
from utils.print_utils import Symbols

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Working on {device} {Symbols.OK}")

    load_zipfile()

    train_loader, val_loader, test_loader, nb_classes = load_dataset("dataset/dataset-train", "dataset/dataset-test")
    inception_v3 = load_inception_for_tl(nb_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(inception_v3.AuxLogits.fc.parameters(), lr=0.001, momentum=0.9)

    print("Apprentissage en transfer learning")
    inception_v3.to(device)
    inception_v3.train(True)
    torch.manual_seed(42)

    train_model(inception_v3, train_loader, val_loader, optimizer, criterion, evaluate, 65, device=device)
    # train_model(inception_v3, criterion, optimizer, None, num_epochs=65, is_inception=True)
    inception_v3.train(False)
    loss, accuracy = evaluate(inception_v3, test_loader, device, criterion)
    print("Accuracy (test): %.1f%%" % (100 * accuracy))

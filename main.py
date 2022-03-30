import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import model.model as model
from model.evaluate import evaluate
from utils.dataset import load_dataset
from utils.print_utils import Symbols


def train_model(model, loader_train, data_val, optimizer, criterion, evaluate, n_epochs=10, device='cpu',
                notebook=False):
    for epoch in range(n_epochs):  # à chaque epochs
        with tqdm(enumerate(loader_train), desc=f"Epoch {epoch}") as pbar:
            for i, data in pbar:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # on passe les données sur CPU / GPU
                optimizer.zero_grad()  # on réinitialise les gradients
                outputs, _ = model(inputs)  # on calcule l'output

                loss = criterion(outputs, labels)  # on calcule la loss
                loss_val, accuracy = evaluate(model, data_val,device, criterion )
                pbar.set_postfix(**{"loss train":loss.item(), "loss val": loss_val, "Acc (val)": accuracy})

                loss.backward()  # on effectue la backprop pour calculer les gradients
                optimizer.step()


# def train_model(model, criterion, optimizer, scheduler, num_epochs=2, is_inception=False):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     model.train()  # Set model to training mode
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase  # Set model to evaluate mode
#
#         running_loss = 0.0
#         running_corrects = 0
#
#         # Iterate over data.
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(True):
#                 # Get model outputs and calculate loss
#                 # Special case for inception because in training it has an auxiliary output. In train
#                 # mode we calculate the loss by summing the final output and the auxiliary output
#                 # but in testing we only consider the final output.
#                 if is_inception:
#                     # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
#                     outputs, aux_outputs = model(inputs)
#                     loss1 = criterion(outputs, labels)
#                     loss2 = criterion(aux_outputs, labels)
#                     loss = loss1 + 0.4*loss2
#                 else:
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#
#                 _, preds = torch.max(outputs, 1)
#
#                 # backward + optimize only if in training phase
#                 loss.backward()
#                 optimizer.step()
#
#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#
#         epoch_loss = running_loss / 5500
#         epoch_acc = running_corrects.double() / 5500
#
#         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#             "train", epoch_loss, epoch_acc))
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Working on {device} {Symbols.OK}")

    train_loader, val_loader, test_loader, nb_classes = load_dataset()
    inception_v3 = model.load_inception_for_tl(nb_classes, device)
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

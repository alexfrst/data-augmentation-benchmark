import torch
from sklearn.metrics import confusion_matrix
def evaluate(model, dataset, device, criterion):
    avg_loss = 0.
    avg_accuracy = 0
    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        batch_acc = torch.sum(preds == labels)/len(preds)

        avg_loss += loss.item()
        avg_accuracy += batch_acc

    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)

def get_confusion_matrix(model, dataset, device):
    mat = None

    classes = list(range(max([max(data[1]) for data in dataset])+1))

    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(torch.sum(preds == labels)/len(preds))
        if mat is None:
            mat = confusion_matrix(labels, preds,labels=classes)
        else:
            mat += confusion_matrix(labels, preds, labels=classes)
    return mat

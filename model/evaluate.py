import torch
def evaluate(model, dataset, device, criterion):
    avg_loss = 0.
    avg_accuracy = 0
    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        batch_acc = torch.sum(preds == labels)/len(preds)

        avg_loss += loss.item()
        avg_accuracy += batch_acc

    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)

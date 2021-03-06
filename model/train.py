from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm
import copy
import torch


def train_model(model, loader_train, data_val, optimizer, scheduler, criterion, evaluate, n_epochs=10, device='cpu', tb_writer=None,training_name='train'):
    best_val_score = 0
    best_params = copy.deepcopy(model.state_dict())
    loss_val, accuracy = 9999,0
    model.train(True)
    for epoch in range(n_epochs):  # à chaque epochs
        with tqdm(enumerate(loader_train), desc=f"Epoch {epoch}") as pbar:
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)  # on passe les données sur CPU / GPU
                optimizer.zero_grad()  # on réinitialise les gradients

                if model._get_name() == "Inception3":
                    outputs, aux_outputs = model(inputs)  # on calcule l'output
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)  # on calcule l'output
                    loss = criterion(outputs, labels)

                pbar.set_postfix(**{"loss train": loss.item(), "loss val": loss_val, "Acc (val)": accuracy})
                if tb_writer is not None:
                    tb_writer.add_scalar(f'{training_name}_Loss/train', loss.item(), epoch)
                    tb_writer.add_scalar(f'{training_name}_Loss/val', loss_val, epoch)
                    tb_writer.add_scalar(f'{training_name}_Accuracy/val', accuracy, epoch)
                    tb_writer.add_scalar(f'{training_name}_LR', optimizer.param_groups[0]['lr'], epoch)

                loss.backward()  # on effectue la backprop pour calculer les gradients
                optimizer.step()
        scheduler.step()
        model.train(False)
        loss_val, accuracy = evaluate(model, data_val, device, criterion)
        pbar.set_postfix(**{"loss train": loss.item(), "loss val": loss_val, "Acc (val)": accuracy})
        model.train(True)
        if accuracy > best_val_score:
            best_val_score = accuracy
            best_params = copy.deepcopy(model.state_dict())
            torch.save(best_params, f'best_params_{training_name}.pt')

    if tb_writer is not None:
        tb_writer.flush()
    model.load_state_dict(torch.load(f'best_params_{training_name}.pt'))
    return model, best_val_score

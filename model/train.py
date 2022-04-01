from tqdm import tqdm


def train_model(model, loader_train, data_val, optimizer, criterion, evaluate, n_epochs=10, device='cpu',
                notebook=False):
    model.train(True)
    for epoch in range(n_epochs):  # à chaque epochs
        with tqdm(enumerate(loader_train), desc=f"Epoch {epoch}") as pbar:
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)  # on passe les données sur CPU / GPU
                optimizer.zero_grad()  # on réinitialise les gradients
                outputs, aux_outputs = model(inputs)  # on calcule l'output
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
                model.train(False)
                loss_val, accuracy = evaluate(model, data_val, device, criterion)
                model.train(True)
                pbar.set_postfix(**{"loss train": loss.item(), "loss val": loss_val, "Acc (val)": accuracy})

                loss.backward()  # on effectue la backprop pour calculer les gradients
                optimizer.step()

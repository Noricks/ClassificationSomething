import torch
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import accuracy_score


# %%
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    predict_all = torch.Tensor([]).to(device)
    labels_all = torch.Tensor([]).to(device)
    model.train()
    for images, labels in dataloader:
        # load to device
        images, labels = images.to(device), labels.to(device)
        # forward
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        # back propagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        # choose the class with the largest possibility
        scores, predictions = torch.max(output.data, 1)
        # append one batch pre and labels to list
        predict_all = torch.cat([predict_all, predictions])
        labels_all = torch.cat([labels_all, labels])

    # get metrics
    train_loss = train_loss / len(dataloader.sampler)
    # retrieve data from gpu
    predictions_cpu = predict_all.cpu()
    labels_cpu = labels_all.cpu()
    train_acc = accuracy_score(labels_cpu, predictions_cpu, normalize=True)
    return train_loss, train_acc


# %%

def val_epoch(model, device, dataloader, loss_fn):
    val_loss = 0.0
    predict_all = torch.Tensor([]).to(device)
    labels_all = torch.Tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            # load to device
            images, labels = images.to(device), labels.to(device)
            # forward
            output = model(images)
            loss = loss_fn(output, labels)
            val_loss += loss.item() * images.size(0)
            # choose the class with the largest possibility
            scores, predictions = torch.max(output.data, 1)
            # append one batch pre and labels to list
            predict_all = torch.cat([predict_all, predictions])
            labels_all = torch.cat([labels_all, labels])

        # get metrics
        val_loss = val_loss / len(dataloader.sampler)
        # retrieve data from gpu
        predictions_cpu = predict_all.cpu()
        labels_cpu = labels_all.cpu()
        val_acc = accuracy_score(labels_cpu, predictions_cpu, normalize=True)
    return val_loss, val_acc


# %%
def test_epoch(model, device, dataloader, loss_fn):
    return val_epoch(model, device, dataloader, loss_fn)

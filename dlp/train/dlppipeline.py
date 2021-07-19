import torch
import numpy as np
import random
import hydra

from omegaconf import DictConfig, OmegaConf

from dlp.data.dataloader import get_mnist, get_cifar10


def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    accuracy = []

    for batch, (images, labels) in enumerate(dataloader):
        # Forward pass:
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running training accuracy for visualisation:
        #predictions = pred.argmax(1)
        #num_correct = (predictions == labels).sum()
        #running_training_acc = float(num_correct) / float(images.shape[0])
        #accuracy.append(running_training_acc)

        # Some metrics:
        #num_iter = epoch * len(dataloader) + batch
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Tensorboard:
            #tb.add_scalar("Training loss per 100 batches", loss, global_step = num_iter)
            #tb.add_scalar("Training accuracy per 100 batches", running_training_acc, global_step = num_iter)


def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")


def train_model(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)

    loss_fn = torch.nn.CrossEntropyLoss()

    #optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    #opt_params = dict(cfg.optimizer)
    #opt_params['params'] = model.parameters()
    #optimizer = init_obj(cfg.optimizer, opt_params)

    train_data, test_data = get_cifar10(batch_size=cfg.batch_size)

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_data, model, loss_fn, optimizer)

    #if (epoch + 1) % 20 == 0:
        #curr_lr /= 3
        #update_lr(optimizer, curr_lr)

        test_loop(test_data, model, loss_fn)

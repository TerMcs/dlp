import torch
import hydra
import socket
# from datetime import datetime
# from pathlib import Path
# import psutil
# import platform


from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from dlp.utils.common import init_obj, get_logger

log = get_logger(__name__)
device = torch.device('cuda:0') #if torch.cuda.is_available() else 'cpu')

#now = datetime.now()
#dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
#log_dir = Path(f'dlp_{dt_string}')
#log_dir.mkdir(parents=True, exist_ok=True)

#tbdir = log_dir / "tb" / "logs"
#tbdir.mkdir(parents=True, exist_ok=True)
tb = SummaryWriter("./runs")

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    accuracy = []

    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device) # this works for first epoch then goes back to CPU -
                                   # this is an issue with the strcuture of the code

        # Forward pass:
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running training accuracy for visualisation:
        predictions = pred.argmax(1)
        num_correct = (predictions == labels).sum()
        running_training_acc = float(num_correct) / float(images.shape[0])
        accuracy.append(running_training_acc)

        # Some metrics:
        #num_iter = epoch * len(dataloader) + batch
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            log.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Tensorboard:
            #tb.add_scalar("Training loss per 100 batches", loss, global_step = num_iter)
            #tb.add_scalar("Training accuracy per 100 batches", running_training_acc, global_step = num_iter)
    return loss, running_training_acc

def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    log.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")
    return test_loss, correct

def train_model(cfg: DictConfig):
    log.info(f'Host: {socket.gethostname()}')
    #tb = SummaryWriter("../../../runs")

    log.info('Using {} device'.format(device))

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    log.info(f'Initialized {cfg.model}')

    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()

    # Select optimizer
    opt_params = dict(cfg.optimizer.params)
    opt_params['params'] = model.parameters()
    optimizer = init_obj(cfg.optimizer.cls, opt_params)

    train_data, test_data = hydra.utils.instantiate(cfg.data)

    for epoch in range(cfg.num_epochs):
        log.info(f"Epoch {epoch + 1}\n-------------------------------")
        # loss, accuracy = \
        train_loop(train_data, model, loss_fn, optimizer)

        # Tensorboard:
        #tb.add_scalar("Training loss per epoch", loss, global_step = epoch)
        #tb.add_scalar("Training accuracy per epoch", accuracy, global_step = epoch)
    #if (epoch + 1) % 20 == 0:
        #curr_lr /= 3
        #update_lr(optimizer, curr_lr)

        test_loop(test_data, model, loss_fn)
        # tb.add_scalar("Test loss per epoch", loss, global_step = epoch)
        # tb.add_scalar("Test accuracy per epoch", accuracy, global_step = epoch)

    # uname = platform.uname()
    # log.info(f"System: {uname.system}")
    # log.info(f"Node Name: {uname.node}")
    # log.info(f"Release: {uname.release}")
    # log.info(f"Version: {uname.version}")
    # log.info(f"Processor: {uname.processor}")
    #svmem = psutil.virtual_memory()
    #log.info(f"Total memory: {svmem.total}")
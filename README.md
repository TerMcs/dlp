# dlp
Further deep learning practice. 

## Example A: 

```
:~/dlp$ python run.py model=fc 
                      model.hidden_size=256
                      data=mnist 
                      data.batch_size=128
                      optimizer.cls=torch.optim.Adam
                      optimizer.params.lr=0.001
                      num_epochs=2
```

### Hydra record:

model:  
&nbsp;&nbsp;_target_: dlp.models.fullyconnected.FullyConnected  
&nbsp;&nbsp;input_size: 784  
&nbsp;&nbsp;hidden_size: 256  
&nbsp;&nbsp;num_classes: 10  
data:  
  _target_: dlp.data.dataloader.get_mnist  
  batch_size: 128  
optimizer:  
  cls: torch.optim.Adam  
  params:  
    params: null  
    lr: 0.001  
seed: 12345  
num_epochs: 2  


## Example B:

```
:~/dlp$ python run.py model=resnet
                      model.depth=56
                      data=cifar10 
                      data.batch_size=64
                      optimizer.cls=torch.optim.SGD
                      optimizer.params.lr=0.01
                      num_epochs=10
```

### Hydar record:

model:  
  _target_: dlp.models.resnet.resnet  
  depth: 56  
data:  
  _target_: dlp.data.dataloader.get_cifar10  
  batch_size: 50  
optimizer:  
  cls: torch.optim.SGD  
  params:  
    params: null  
    lr: 0.01  
seed: 12345  
num_epochs: 10  



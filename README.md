# Improving Line Search Methods for Large Scale Neural Network Training

The Repository to the Paper Improving Line Search Methods for Large Scale Neural Network Training

More recent implementation at:

https://github.com/TheMody/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation
## Abstract

In recent studies, line search methods have shown significant improvements in the performance of traditional stochastic gradient descent techniques, eliminating the need for a specific learning rate schedule \cite{vaswani20a, mahsereci15a, vaswani2021adaptive}. In this paper, we identify existing issues in state-of-the-art line search methods \cite{vaswani20a, vaswani2021adaptive}, propose enhancements, and rigorously evaluate their effectiveness. We test these methods on larger datasets and more complex data domains than before.

Specifically, we improve the Armijo line search by integrating the momentum term from ADAM in its search direction, enabling efficient large-scale training, a task that was previously prone to failure using Armijo line search methods. Our optimization approach outperforms both the previous Armijo implementation and tuned learning rate schedules for Adam.

Our evaluation focuses on Transformers and CNNs in the domains of NLP and image data.

Our work is publicly available as a Python package, which provides a hyperparameter free Pytorch optimizer.

## Install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3

for replication:
- `pip install transformers` for huggingface transformers <3 
- `pip install datasets` for huggingface datasets <3 
- `pip install tensorflow-datasets` for tensorflow datasets <3 
- `pip install wandb` for optional logging <3
- for easy replication use conda and environment.yml eg:
`$ conda env create -f environment.yml` and `$ conda activate sls3`



## Use in own projects

The custom optimizer is in \sls\ken_sls.py and \sls\ken_base.py 
Example Usage:

```
from sls.adam_sls import KenSLS
optimizer = KenSLS([model.parameters()])
```

The typical pytorch forward pass needs to be changed from :
``` 
optimizer.zero_grad()
y_pred = model(batch_x)
loss = criterion(y_pred, batch_y)    
loss.backward()
optimizer.step()
scheduler.step() 
```
to:
``` 
closure = lambda : criterion(model(batch_x), batch_y)
optimizer.zero_grad()
loss = optimizer.step(closure = closure)
```

This code change is necessary since, the optimizers needs to perform additional forward passes and thus needs to have the forward pass encapsulated in a function.
see embedder.py in the fit() method for more details


## Replicating Results
For replicating the main Results of the Paper run:

```
$ python run_multiple.py
```

and 

```
$ python run_multiple_img.py
```


For replicating specific runs or trying out different hyperparameters use:

```
$ python main.py 
```

and change the config.json file appropriately



## Please cite:
Improving Line Search Methods for Large Scale Neural Network Training
with Line Search Methods 
from 
Philip Kenneweg,
Tristan Kenneweg,
Barbara Hammer
To be published in ACDSA 2024

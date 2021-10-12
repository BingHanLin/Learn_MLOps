# PyTorch Lightning Introduction

## Install

```
pip install pytorch-lightning
```

## Basic Concept

In PyTorch Lightning, the codes are organized in the following structure.

* **Research code**: 
    The core framework which is designed based on each task. User can define them by deriving a subclass from the **LightningModule**.

* **Engineering code**: 
    The codes that can be found cross various tasks. They are handled by the **Trainer**.

* **Non-essential research code**: 
    Including logging, visulization, etc... this goes in **Callbacks**.

* **Data**:
    Use PyTorch DataLoaders or organize them into a **LightningDataModule**.

## LightningModule (nn.Module subclass)
Build your own system by deriving a subclass from the **LightningModule**

``` python
import pytorch_lightning as pl

class myNet(pl.LightningModule):

    def somemethod(self):
        pass
```
A LightningModule organizes your PyTorch code into 5 sections:
* Computations (init).
* Train loop (training_step)
* Validation loop (validation_step)
* Test loop (test_step)
* Optimizers (configure_optimizers)

To train the system automatically, some methods need to be overrided:

* forward(*args, **kwargs)
    Define the process in the forward propagation.

``` python
model = TestModel()
x = torch.Tensor([1, 2, 3])
output1 = model(x)
output2 = model.forward(x) # Same as above.
```

* prepare_data()


* configure_optimizers()

* training_step/val_step/test_step(*args, **kwargs)
    Define the process in the each step of traing/validation/test.

* training/validation/test_epoch_end(outputs)





## Reference:

* https://medium.com/%E6%95%B8%E5%AD%B8-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E8%88%87%E8%9F%92%E8%9B%87/pytorch-lightning-%E5%85%A5%E5%9D%91%E5%BF%83%E5%BE%97-81af12de9bb7

* https://minglunwu.github.io/notes/2020/20200416.html

* https://github.com/PyTorchLightning/pytorch-lightning

* https://pytorch-lightning.readthedocs.io/en/latest/index.html

* [A PyTorch Lightning Template](https://zhuanlan.zhihu.com/p/353985363)
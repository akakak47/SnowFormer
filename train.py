import pytorch_lightning as pl  # 导入PyTorch Lightning库，一个用于简化PyTorch训练过程的库
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
from pytorch_lightning.strategies import DDPStrategy  # 从PyTorch Lightning库中导入分布式数据并行策略
from pytorch_lightning.callbacks import ModelCheckpoint  # 从PyTorch Lightning库中导入模型检查点回调
from dataloader import *  # 从dataloader模块导入所有内容（通常是数据加载相关的功能）
# from ema import *  # 注释掉了从ema模块导入所有内容的语句，可能是暂时不需要或者ema模块不存在
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的神经网络功能模块
from copy import deepcopy  # 导入Python的深拷贝功能
import torchvision  # 导入torchvision库，通常用于加载预训练的模型和图像变换
import torchvision.transforms as transforms  # 导入torchvision的图像变换模块
from loss.CL1 import L1_Charbonnier_loss, PSNRLoss  # 从自定义的loss模块中导入L1_Charbonnier_loss和PSNRLoss损失函数
from loss.perceptual import PerceptualLoss2  # 从自定义的loss模块中导入PerceptualLoss2感知损失函数
from argparse import Namespace  # 导入argparse库中的Namespace类，用于创建命名空间对象

from pytorch_lightning import seed_everything  # 从PyTorch Lightning库中导入全局种子设置函数
from metrics import PSNR, SSIM  # 从metrics模块中导入PSNR和SSIM评估指标
from D2PBM_Net import Transformer  # 从D2PBM_Net模块中导入Transformer模型

seed = 42  # 设置全局种子为42，确保实验结果的可复现性
seed_everything(seed)  # 使用之前导入的全局种子设置函数设置种子

from pytorch_lightning.loggers import TensorBoardLogger  # 从PyTorch Lightning库中导入TensorBoard日志记录器
import torch.multiprocessing  # 导入PyTorch的多进程模块

torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享策略为'file_system'
import tensorboardX  # 导入tensorboardX库，用于可视化训练过程

wandb_logger = TensorBoardLogger(r'/home/csx/data/DSR_ACMMM/tb_logs',
                                 name='SnowFormer')  # 创建一个TensorBoard日志记录器，指定日志保存路径和实验名称

img_channel = 3  # 定义图像通道数为3，通常表示RGB图像
width = 32  # 定义宽度为32，可能是模型或图像处理中某个维度的参数
img_size = 256  # 定义图像大小为256x256


class EMA(nn.Module):  # 定义一个EMA（指数移动平均）类，继承自PyTorch的nn.Module
    """ Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):  # 初始化函数，接收一个模型和衰减率作为参数
        super(EMA, self).__init__()  # 调用父类nn.Module的初始化方法
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)  # 创建一个模型的深拷贝，用于累积权重的移动平均
        self.module.eval()  # 将模型设置为评估模式
        self.decay = decay  # 设置衰减率

    def _update(self, model, update_fn):  # 定义一个内部更新函数，接收模型和更新函数作为参数
        with torch.no_grad():  # 不计算梯度
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))  # 使用更新函数更新EMA模型的权重

    def update(self, model):  # 定义一个更新函数，接收模型作为参数
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)  # 调用内部更新函数，并传入一个特定的更新函数

    def set(self, model):  # 定义一个设置函数，接收模型作为参数
        self._update(model, update_fn=lambda e, m: m)  # 调用内部

class CoolSystem(pl.LightningModule): # 定义一个名为CoolSystem的类，继承自PyTorch Lightning的LightningModule

    def __init__(self, hparams):  # 初始化函数，接收一个包含超参数的命名空间对象作为参数
        super(CoolSystem, self).__init__()  # 调用父类LightningModule的初始化方法

        self.params = hparams  # 将传入的超参数保存到类的实例变量中
        
        # train/val/test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs

        #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers

        #loss_function
        self.loss_f = PSNRLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_per = PerceptualLoss2()
        self.model = Transformer()

    def forward(self, x):
        y = self.model(x)
        #self.ema.update() 
        return y
    
    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr,betas=[0.9,0.999])
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr,max_lr=1.2*self.initlr,cycle_momentum=False)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y,_= batch

        output = self.forward(x)
        loss_f = self.loss_f(y,output)
        loss_per = self.loss_per(y,output)

        loss = (loss_f + 0.2*loss_per )
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y,_ = batch

        y_hat,_= self.forward(x)
        loss = self.loss_f(y,y_hat[0]) + 0.2*self.loss_per(y,y_hat[0])# + loss_uncertarinty
        psnr = PSNR(y_hat[0],y)
        ssim = SSIM(y_hat[0],y)
        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)
        
        self.trainer.checkpoint_callback.best_model_score #save the best score model
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_image('input',x[0],0)
            tensorboard.add_image('gt',y[0],0)
        return {'val_loss': loss, 'psnr': psnr,'ssim':ssim}


    def train_dataloader(self):
        # REQUIRED
        train_set = CSD_Dataset(self.train_datasets,train=True,size=self.crop_size)
        #train_set = RealWorld_Dataset(self.train_datasets,train=True,size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
    
    def val_dataloader(self):
        val_set = CSD_Dataset(self.validation_datasets,train=False,size=256)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=True, num_workers=self.num_workers)
        
        return val_loader
def main():
    resume = False,
    resume_checkpoint_path = None
    
    args = {
    'epochs': 10000,
    'train_datasets':r'/home/csx/data/CSD/Train',
    'test_datasets':None,
    'val_datasets':r'/home/csx/data/CSD/Test',

    #bs
    'train_bs':12,
    'test_bs':40,
    'val_bs':40,
    'initlr':0.0006,
    'weight_decay':0.01,
    'crop_size':256,#128
    'num_workers':16,
    #Net
    'model_blocks':5,
    'chns':64
    }
 
    ddp = DDPStrategy(find_unused_parameters=True)
    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    #dirpath='/mnt/data/yt/Documents/TSANet-underwater/snapshots',
    filename='CSD-v5-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=10,
    mode = "max"
    )

    if resume==True:
        trainer = pl.Trainer(
            strategy = ddp,
            max_epochs=hparams.epochs,
            devices = [0,1,2,3,4,5,6],
            logger=wandb_logger,
            #amp_backend="apex",
            #amp_level='01',
            #accelerator='ddp',
            #precision=16,
            callbacks = [checkpoint_callback],
        ) 
    else:
        trainer = pl.Trainer(
            strategy = ddp,
            max_epochs=hparams.epochs,
            devices = [0,1,2,3,4,5,6],
            logger=wandb_logger,
            #amp_backend="apex",
            #amp_level='01',
            #accelerator='ddp',
            #precision=16,
            callbacks = [checkpoint_callback],
        )  

    trainer.fit(model,ckpt_path=resume_checkpoint_path)

if __name__ == '__main__':
	#your code
    main()
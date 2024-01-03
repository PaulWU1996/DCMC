import torch
import torch.nn as nn
import pytorch_lightning as pl

from cores.model.DCMC import DCMC
from cores.training.loss import CustomLoss
from cores.training.metrics import eval

from utils.data import DataModule

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from pytorch_lightning.tuner import Tuner

class CustomEarlyStopping(Callback):
    def __init__(self, monitor_metrics, thresholds, patience):
        super(CustomEarlyStopping, self).__init__()
        self.monitor_metrics = monitor_metrics
        self.thresholds = thresholds
        self.patience = patience
        self.counter = 0

    def on_validation_end(self, trainer, pl_module):
        monitor_values = {metric: trainer.callback_metrics.get(metric, float('inf')) for metric in self.monitor_metrics}
        
        # Check if accuracy is highest and distance is minimized
        accuracy_condition = monitor_values['eval_accuracy'] >= self.thresholds['eval_accuracy']
        distance_condition = monitor_values['eval_loss'] <= self.thresholds['eval_loss']
        
        if accuracy_condition and distance_condition:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
        else:
            self.counter = 0

class MyCallback(Callback):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.patience = config["training"]["patience"]

        self.counter = 0
        self.best_state = {'distance': float('inf'), 'accuracy': 0.0}
        self.ckpt_path = config["filesystem"]["root"] + config["filesystem"]["result_dir"]


    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
        print("The model structure:\n {}".format(pl_module))

    def on_train_end(self, trainer, pl_module):
        print("Training is finished!")

    def on_validation_epoch_end(self, trainer, pl_module):
        distance = torch.stack(pl_module.validatation_step_outputs['eval_loss']).mean()
        total = sum(pl_module.validatation_step_outputs['total'])
        correct = sum(pl_module.validatation_step_outputs['correct'])
        accuracy = correct / total

        if distance < self.best_state['distance'] or accuracy > self.best_state['accuracy']:
            self.best_state['distance'] = distance
            self.best_state['accuracy'] = accuracy
            self.counter = 0
            filename = self.ckpt_path +  '/best_model.ckpt'
            # "/best_distance: {:.4f}, best_accuracy: {:.4f}.ckpt".format(self.best_state['distance'], self.best_state['accuracy'])
            trainer.save_checkpoint(filename)
            print("\nCurrent Best model: ( distance: {:.4f}, accuracy: {:.4f})".format(self.best_state['distance'], self.best_state['accuracy']))
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True



    
    

def train(config):
    # torch setting for A100 GPU
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')

    # set logger
    # wandb_logger = WandbLogger(project='DCMC', entity='', log_model=True, save_dir=config["filesystem"]["root"] + config["filesystem"]["log_dir"])

    # load model and data
    model = DCMC(config)
    data = DataModule(config)

    # # define custom callback
    # monitor_metrics = ['eval_accuracy', 'eval_loss']
    # thresholds = {'eval_accuracy': 0.9, 'eval_loss': 0.02}
    # early_stopping = CustomEarlyStopping(monitor_metrics, thresholds, patience=10)
    mycallback = MyCallback(config)


    trainer = pl.Trainer(
        accelerator='auto',
        default_root_dir=config["filesystem"]["root"] + config["filesystem"]["log_dir"],
        min_epochs=3,
        max_epochs=config["training"]["max_epoch"],
        check_val_every_n_epoch=1,
        callbacks=[mycallback],
        # logger=wandb_logger,
        fast_dev_run=True,
    )

    # if config["training"]["auto_lr_find"]:
    if config["training"]["auto_lr_find"]:
        tuner = Tuner(trainer=trainer)
        lr_finder = tuner.lr_find(model, data)
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print("New learning rate: {}".format(new_lr))
    trainer.fit(model, data)
    

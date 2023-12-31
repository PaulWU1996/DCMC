# import sys
# sys.path.append('/vol/research/VS-Work/PW00391/D-CMCM')

from cores.model.cmconformer import Conformer, CMConformer

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from cores.training.loss import CustomLoss
from cores.training.metrics import eval

import pytorch_lightning as pl

class FeatureExtractor(nn.Module):
    def __init__(self,):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=512, out_features=128, bias=True)
        self.resnet = resnet
        # self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # self.conv1d1 = nn.Conv1d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.b1 = nn.BatchNorm1d(256)
        # self.prelu = nn.PReLU()
        # self.conv1d2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet(x)
        x = x.unsqueeze(1) # (batch_size, 1, 128) -> (batch_size, timestep=1, dim=128)

        # x = x.flatten(2)
        # x = self.conv1d1(x)
        # x = self.b1(x)
        # x = self.prelu(x)
        # x = self.conv1d2(x)
        # x = x.transpose(1, 2)
        return x 

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TargetPredictor(nn.Module):
    def __init__(self, in_dim, num_targets):
        super().__init__()
        # only predict the location of the target atm
        self.num_targets = num_targets
        self.loc_predictor = MLP(input_dim=in_dim, hidden_dim=in_dim, output_dim=2*num_targets, num_layers=2)
        # self.cls_predictor = MLP(input_dim=in_dim, hidden_dim=in_dim, output_dim=num_targets, num_layers=2) 

    def forward(self, x):
        # output_cls = self.cls_predictor(x)
        output_loc = self.loc_predictor(x).sigmoid()
        return output_loc.view(-1, self.num_targets, 2) # output_cls.view(-1, self.num_targets),
    
class DCMC(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # params
        self.config = config
        self.threshold = config['training']['threshold'] / config['settings']['scenario_x']
        self.lr = config['training']['learning_rate']

        # network
        self.extractor = FeatureExtractor()
        # TODO: instantiate the conformer or cmconformer based on config
        if config['encoder']['type'] == 'Conformer':
            self.encoder = Conformer(in_dim=config['encoder']['in_dim'],ffn_dim=config['encoder']['ffn_dim'],num_heads=config['encoder']['num_heads'],num_layers=config['encoder']['num_layers'],depthwise_kernel_size=config['encoder']['depthwise_kernel_size'],dropout=config['encoder']['dropout'],use_gn=config['encoder']['use_gn'],conv_first=config['encoder']['conv_first'])
        elif config['encoder']['type'] == 'CMConformer':
            self.encoder = CMConformer(in_dim=config['encoder']['in_dim'],ffn_dim=config['encoder']['ffn_dim'],num_heads=config['encoder']['num_heads'],num_layers=config['encoder']['num_layers'],depthwise_kernel_size=config['encoder']['depthwise_kernel_size'],dropout=config['encoder']['dropout'],use_gn=config['encoder']['use_gn'],conv_first=config['encoder']['conv_first'])
        elif config['encoder']['type'] == 'None':
            self.encoder = None
        else:
            raise NotImplementedError
        
        self.predictor = TargetPredictor(in_dim=config['encoder']['in_dim'], num_targets=config['settings']['target_num_max'])

        # training
        self.train_loss = nn.MSELoss()
        self.val_loss = CustomLoss()
        self.validatation_step_outputs = {'eval_loss': [], 'total': [], 'correct': []} # [loss, total, correct]


    def forward(self, input:torch.Tensor, second_input:torch.Tensor=None):
        """
        Args:
            input: (batch_size, 1, 512, 512)
            second_input: None or (batch_size, 1, 512, 512)
        """
        x = self.extractor(input)
        if second_input is not None:
            y = self.extractor(second_input)
            if self.encoder is not None:
                x, _ = self.encoder(input=x, second_input=y)
        else:
            if self.encoder is not None:
                x, _ = self.encoder(x)
        output_loc = self.predictor(x)
        return output_loc

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            input, target = batch
            second_input = None
        else:
            input, second_input, target = batch #TODO: update MultiDataset and return [input, second_input, target]
        output_loc = self(input=input, second_input=second_input)
        loss = self.train_loss(output_loc, target)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            input, target = batch
            second_input = None
        else:
            input, second_input, target = batch #TODO: update MultiDataset and return [input, second_input, target]
        output_loc = self(input=input, second_input=second_input)
        loss = self.val_loss(output_loc, target)
        correct, total = eval(output_loc, target, self.threshold)
        self.validatation_step_outputs['eval_loss'].append(loss)
        self.validatation_step_outputs['total'].append(total)
        self.validatation_step_outputs['correct'].append(correct)

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validatation_step_outputs['eval_loss']).mean()
        total = sum(self.validatation_step_outputs['total'])
        correct = sum(self.validatation_step_outputs['correct'])

        self.log('eval_distance', loss, prog_bar=True, logger=True)
        self.log('eval_accuracy', correct / total, prog_bar=True, logger=True)

        self.validatation_step_outputs = {'eval_loss': [], 'total': [], 'correct': []} # reset the memory and keep [loss, total, correct]

    def configure_optimizers(self):

        # freeze the extractor
        # for param in self.extractor.parameters():
        #     param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [scheduler] 
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-6)
        # return {'optimizer': optimizer, 'scheduler':scheduler, 'monitor':'eval_distance'}

    
        

    
        

# input_data = torch.randn(2, 1, 512, 512)
# extractor = FeatureExtractor()
# output = extractor(input_data)
# conformer = Conformer(in_dim=128,ffn_dim=256,num_heads=4,num_layers=2,depthwise_kernel_size=1,dropout=0.1)
# cmconformer = CMConformer(in_dim=128,ffn_dim=256,num_heads=4,num_layers=2,depthwise_kernel_size=1,dropout=0.1)
# x, _ = conformer(output)
# predictor = TargetPredictor(in_dim=128, num_targets=3)
# out_loc = predictor(x)

# print(output.shape)



# input_data = torch.randn(2, 1, 512, 512)
# # resnet = FeatureExtractor()
# # output = resnet(input_data)
# print(output.shape)

# # considering the block directly , we can complete everthing tomorrow morning!

# # conformer = Conformer(in_dim=128,ffn_dim=256,num_heads=4,num_layers=2,depthwise_kernel_size=1,dropout=0.1)
# # cmconformer = CMConformer(in_dim=128,ffn_dim=256,num_heads=4,num_layers=2,depthwise_kernel_size=1,dropout=0.1)
# # x, _ = conformer(output)
# # predictor = TargetPredictor(in_dim=128, num_targets=3)
# # out_loc = predictor(x)
# DCMC
# loss = CustomLoss()
# print(loss(out_loc[0], out_loc[1]))
# # metric = CustomMetric()
# print(eval(out_loc[0], out_loc[1],0.13))
# print(conformer)







    
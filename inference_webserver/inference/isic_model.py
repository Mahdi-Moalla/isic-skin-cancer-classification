import pytorch_lightning as pl   ########

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class isic_classifier(pl.LightningModule):

    output_features=2

    def __init__(self,
                 config,
                 train_mode='transfer-learning'):

        super().__init__()
        
        self.config=config
        if self.config.pretrained_weights_type=="torchvision_id":
            self.model=torchvision.models.get_model(self.config.model,
                                                    weights=self.config.pretrained_weights)
        elif self.config.pretrained_weights_type=="path":
            self.model=torchvision.models.get_model(self.config.model, weights=None)
            if  self.config.pretrained_weights is not  None:
                self.model.load_state_dict(torch.load(self.config.pretrained_weights))
        else:
            raise Exception(f'unknown pretrained_weights_type {self.config.pretrained_weights_type}')

        self.train_mode=train_mode
        
        #######################################################
        # generic update of classification layer
        
        transfer_learning_layer=self.config.transfer_learning_layer

        ## finding last-1 layer  
        layer=self.model
        
        for layer_id in transfer_learning_layer[:-1]:
            if  type(layer_id)  is str:
                layer=getattr(layer,layer_id)
            elif type(layer_id)  is int:
                layer=layer[layer_id]
            else:
                raise Exception(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")


        ## updating last layer

        #self.model.new_fc_layer=nn.Sequential(*[nn.LazyLinear(out_feats) \
        #                    for  out_feats in self.config.transfer_learning_layer_spec])

        self.model.new_fc_layer=nn.ModuleDict({'transfer_learning':
                nn.Sequential(*[nn.LazyLinear(out_feats) \
                            for  out_feats in self.config.transfer_learning_layer_spec[:-1]]),
                                              'merger':nn.LazyLinear(
                                                self.config.transfer_learning_layer_spec[-1])})

        layer_id=transfer_learning_layer[-1]
        if  type(layer_id)  is str:
            
            curr_last_layer  =  getattr(layer,layer_id) 
            assert type( curr_last_layer ) is nn.Linear
            in_features = curr_last_layer.in_features
            #self.new_fc_layer=nn.Linear(in_features , self.output_features)
            
            #layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            #self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])
            
            setattr(layer,layer_id, self.model.new_fc_layer['transfer_learning'] )
            
        elif type(layer_id)  is int:
            curr_last_layer  =  layer[layer_id] 
            assert type( curr_last_layer ) is nn.Linear
            in_features = curr_last_layer.in_features
            #self.new_fc_layer=nn.Linear(in_features , self.output_features )
            
            #layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            #self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])
            
            layer[layer_id] =  self.model.new_fc_layer['transfer_learning']
        else:
            raise Exception(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")

        

    def forward(self, x, tab_feats):
        x= self.model(x)
        #import  pdb; pdb.set_trace()
        x=self.model.new_fc_layer['merger'](torch.cat([x,tab_feats],dim=1))
        #import  pdb; pdb.set_trace()
        return x

    def training_step(self, batch, batch_idx):
        #import  pdb; pdb.set_trace()
        x, tab_feats, y = batch
        #import  pdb; pdb.set_trace()
        logits = F.log_softmax(self(x,tab_feats ), dim=1)
        
        loss = F.nll_loss(logits, y,
                          weight = torch.tensor([1.0,self.config.pos_class_weight], 
                                                device=self.device))
        self.log("train_loss", loss,  logger=None)
        return loss

    def evaluate(self, batch, stage=None):
        #import  pdb; pdb.set_trace()
        x, tab_feats, y = batch
        #import  pdb; pdb.set_trace()
        logits = F.log_softmax(self(x,tab_feats ), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.validation_targets.extend(y.tolist())
        self.validation_scores.extend(F.softmax(logits)[:,1].tolist())
        
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        x, tab_feats,_=batch
        return F.softmax(self(x, tab_feats), dim=1)[...,1]

    def configure_optimizers(self):

        try:
            optimizer_cls=dynamic_import(self.config.optimizer)
        except:
            print('failed to import optimizer')
            exit(1)

        try:
            lr_scheduler_cls=dynamic_import(self.config.lr_sceduler)
        except:
            print('failed to import lr scheduler')
            exit(1)
        
        
        if self.train_mode=='transfer-learning':
            optimizer = optimizer_cls(self.model.new_fc_layer.parameters(),
                                      lr=self.config.learning_rate,
                                  **self.config.optimizer_params)
        elif self.train_mode=='finetuning':
            optimizer = optimizer_cls(self.parameters(),
                                      lr=self.config.finetune_learning_rate,
                                      **self.config.optimizer_params)
        else:
            raise Exception('unknown train mode')
        
        return {
                "optimizer":optimizer,
                "lr_scheduler": {"scheduler":lr_scheduler_cls(optimizer,
                                                        **self.config.lr_sceduler_params),
                                 "monitor": "train_loss"}
        }



"""
classification pytorch model
"""

# pylint: disable=import-error
import sys

# isort: off
import torch
from torch import nn
import torch.nn.functional as F

# isort: on
import pytorch_lightning as pl
import torchvision


def dynamic_import(name):
    """
    import class by name
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_toplast_layer(model, transfer_learning_layer):
    """
    find top classification  layer in a torchvision
    classification  model
    """
    layer = model

    for layer_id in transfer_learning_layer:  # [:-1]:
        if isinstance(layer_id, str):
            layer = getattr(layer, layer_id)
        elif isinstance(layer_id, int):
            layer = layer[layer_id]
        else:
            raise TypeError(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")
    return layer


class isic_classifier(
    pl.LightningModule
):  # pylint: disable=invalid-name, consider-using-from-import
    """
    pytorch  classification model
    """

    output_features = 2

    def __init__(self, config, train_mode='transfer-learning'):

        super().__init__()

        self.config = config
        if self.config.pretrained_weights_type == "torchvision_id":
            self.model = torchvision.models.get_model(
                self.config.model, weights=self.config.pretrained_weights
            )
        elif self.config.pretrained_weights_type == "path":
            self.model = torchvision.models.get_model(self.config.model, weights=None)
            if self.config.pretrained_weights is not None:
                self.model.load_state_dict(torch.load(self.config.pretrained_weights))
        else:
            raise ValueError(
                f'unknown pretrained_weights_type {self.config.pretrained_weights_type}'
            )

        self.train_mode = train_mode

        #######################################################
        # generic update of classification layer

        transfer_learning_layer = self.config.transfer_learning_layer

        # ## finding last-1 layer
        # layer = self.model

        # for layer_id in transfer_learning_layer[:-1]:
        #     if isinstance(layer_id, str):
        #         layer = getattr(layer, layer_id)
        #     elif isinstance(layer_id, int):
        #         layer = layer[layer_id]
        #     else:
        #         raise TypeError(
        #             f"unknown layer_id type: {type(layer_id)} of  {layer_id}"
        #         )

        layer = get_toplast_layer(self.model, transfer_learning_layer[:-1])

        ## updating last layer

        # self.model.new_fc_layer=nn.Sequential(*[nn.LazyLinear(out_feats) \
        #                    for  out_feats in self.config.transfer_learning_layer_spec])

        self.model.new_fc_layer = nn.ModuleDict(
            {
                'transfer_learning': nn.Sequential(
                    *[
                        nn.LazyLinear(out_feats)
                        for out_feats in self.config.transfer_learning_layer_spec[:-1]
                    ]
                ),
                'merger': nn.LazyLinear(self.config.transfer_learning_layer_spec[-1]),
            }
        )

        layer_id = transfer_learning_layer[-1]
        if isinstance(layer_id, str):

            curr_last_layer = getattr(layer, layer_id)
            assert isinstance(curr_last_layer, nn.Linear)
            # in_features = curr_last_layer.in_features
            # self.new_fc_layer=nn.Linear(in_features , self.output_features)

            # layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            # self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])

            setattr(layer, layer_id, self.model.new_fc_layer['transfer_learning'])

        elif isinstance(layer_id, int):
            curr_last_layer = layer[layer_id]
            assert isinstance(curr_last_layer, nn.Linear)
            # in_features = curr_last_layer.in_features
            # self.new_fc_layer=nn.Linear(in_features , self.output_features )

            # layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            # self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])

            layer[layer_id] = self.model.new_fc_layer['transfer_learning']
        else:
            raise TypeError(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")

    # pylint: disable=arguments-differ
    def forward(self, x, tab_feats):
        """
        model forward method
        """
        x = self.model(x)
        x = self.model.new_fc_layer['merger'](torch.cat([x, tab_feats], dim=1))
        return x

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        """
        training step
        """
        x, tab_feats, y = batch
        logits = F.log_softmax(self(x, tab_feats), dim=1)

        loss = F.nll_loss(
            logits,
            y,
            weight=torch.tensor([1.0, self.config.pos_class_weight], device=self.device),
        )
        self.log("train_loss", loss, logger=None)
        return loss

    def evaluate(self, batch, stage=None):
        """
        evaluation  method
        """
        x, tab_feats, y = batch
        logits = F.log_softmax(self(x, tab_feats), dim=1)
        loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)

        self.validation_targets.extend(y.tolist())
        self.validation_scores.extend(F.softmax(logits)[:, 1].tolist())

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        """
        validation step
        """
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        """
        test step
        """
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # pylint: disable=unused-argument
        """
        prediction step
        """
        x, tab_feats, _ = batch
        return F.softmax(self(x, tab_feats), dim=1)[..., 1]

    def configure_optimizers(self):
        """
        configuring optimizers
        """
        try:
            optimizer_cls = dynamic_import(self.config.optimizer)
        except (ModuleNotFoundError, AttributeError):
            print('failed to import optimizer')
            sys.exit(1)

        try:
            lr_scheduler_cls = dynamic_import(self.config.lr_sceduler)
        except (ModuleNotFoundError, AttributeError):
            print('failed to import lr scheduler')
            sys.exit(1)

        if self.train_mode == 'transfer-learning':
            optimizer = optimizer_cls(
                self.model.new_fc_layer.parameters(),
                lr=self.config.learning_rate,
                **self.config.optimizer_params,
            )
        elif self.train_mode == 'finetuning':
            optimizer = optimizer_cls(
                self.parameters(),
                lr=self.config.finetune_learning_rate,
                **self.config.optimizer_params,
            )
        else:
            raise ValueError('unknown train mode')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler_cls(optimizer, **self.config.lr_sceduler_params),
                "monitor": "train_loss",
            },
        }

"""
pytorch-lightning callback for computing auc metric
and auc(tpr>=0.8) metric per epoch
"""

# pylint: disable=import-error
import os
import os.path as osp

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import auc, roc_auc_score, roc_curve


class binary_auroc_scorer(Callback):  # pylint: disable=invalid-name
    """
    callback class
    """

    def __init__(self, config):
        self.config = config
        self.sanity_check = False
        self.val_binary_auc = []
        self.val_binary_auc_tpr08 = []
        os.makedirs(osp.join(config.log_dir, 'aucs'), exist_ok=True)
        self.train_mode_lut = {'transfer-learning': 0, 'finetuning': 1}

    def on_fit_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        """
        on fit start callback
        """
        self.val_binary_auc = []
        self.val_binary_auc_tpr08 = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        on validation epoch end callback
        """
        if not self.sanity_check:
            self.sanity_check = True
            return

        fold_i_str = f"fold_{trainer.fold_i}_" if trainer.fold_i is not None else ""

        val_scores_np = np.array(pl_module.validation_scores)
        val_targets_np = np.array(pl_module.validation_targets)
        val_weights_np = np.array(
            [
                self.config.roc_pos_class_weight if x == 1 else 1.0
                for x in pl_module.validation_targets
            ]
        )

        fpr, tpr, _ = roc_curve(val_targets_np, val_scores_np)
        fpr_1 = fpr[tpr >= 0.8]
        tpr_1 = tpr[tpr >= 0.8]

        if len(fpr_1) == 0 or len(tpr_1) == 0:
            return

        # print(f'tpr_1[0]={tpr_1[0]}')
        fpr_1 -= fpr_1[0]
        tpr_1 -= tpr_1[0]

        score_08 = auc(fpr_1, tpr_1)

        score = roc_auc_score(val_targets_np, val_scores_np, sample_weight=val_weights_np)

        self.val_binary_auc.append(score)
        self.val_binary_auc_tpr08.append(score_08)

        print(f'@@@@@@@ binary au_roc_tpr.8 = {score_08} @@@@@@@')
        print(f'@@@@@@@ binary au_roc       = {score} @@@@@@@')

        mlflow.log_metric(
            key=f"{pl_module.train_mode} binary au_roc",
            value=score,
            step=trainer.current_epoch,
        )

        mlflow.log_metric(
            key=f"{pl_module.train_mode} binary au_roc_tpr.8",
            value=score_08,
            step=trainer.current_epoch,
        )

        pl_module.log("val_binary_auc_tpr.8", score_08, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("val_binary_auc", score, on_epoch=True, prog_bar=True, logger=True)
        if trainer.fold_i is not None:
            pl_module.log(
                "fold_i",
                trainer.fold_i,
                on_epoch=True,
                # prog_bar =True,
                logger=True,
            )
        pl_module.log(
            "train_mode",
            self.train_mode_lut[pl_module.train_mode],
            on_epoch=True,
            # prog_bar =True,
            logger=True,
        )
        pl_module.log(
            "log_val",
            1,
            on_epoch=True,
            # prog_bar =True,
            logger=True,
        )

        # fpr, tpr, _ = roc_curve(val_targets_np, val_scores_np)

        fig = plt.figure()
        plt.plot(fpr, tpr, color='b')
        plt.plot([0, 1], [0.8, 0.8], color='r')
        plt.title(f'roc at epoch {trainer.current_epoch}')
        # plt.savefig(osp.join(self.config.log_dir,'rocs',
        # f'fold_{trainer.fold_i}_mode_{pl_module.train_mode}
        # _epoch_{trainer.current_epoch}.png'))
        # plt.show()
        mlflow.log_figure(
            fig,
            artifact_file=f'{fold_i_str}mode_{pl_module.train_mode}'
            + f'_epoch_{trainer.current_epoch}.png',
        )

        fig = plt.figure()
        plt.plot(self.val_binary_auc, color='b', marker="*")
        plt.title(f'aucs up to {trainer.current_epoch}')
        # plt.show()
        mlflow.log_figure(
            fig,
            artifact_file=f'{fold_i_str}mode_{pl_module.train_mode}_auc.png',
        )

    def on_fit_end(self, trainer, pl_module):
        """
        on fit end callback
        """

        fold_i_str = f"fold_{trainer.fold_i}_" if trainer.fold_i is not None else ""

        plt.figure()
        plt.plot(self.val_binary_auc, color='b', marker="*")
        plt.title('aucs')
        plt.savefig(
            osp.join(
                self.config.log_dir,
                'aucs',
                f'auc_{fold_i_str}mode_{self.train_mode_lut[pl_module.train_mode]}.png',
            )
        )
        plt.show()

        plt.figure()
        plt.plot(self.val_binary_auc_tpr08, color='r', marker="*")
        plt.title('aucs tpr=.8')
        plt.savefig(
            osp.join(
                self.config.log_dir,
                'aucs',
                f'aucs08_{fold_i_str}mode_' + f'{self.train_mode_lut[pl_module.train_mode]}.png',
            )
        )
        plt.show()

    def on_validation_epoch_start(self, trainer, pl_module):  #  pylint: disable=unused-argument
        """
        on validation epoch end callback
        """
        pl_module.validation_targets = []
        pl_module.validation_scores = []

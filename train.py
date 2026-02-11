from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os


@hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    class OverwriteModelCheckpoint(ModelCheckpoint):
        def _save_checkpoint(self, trainer, filepath):
            self.best_filepath = 'best_model.ckpt'
            if os.path.exists(self.best_filepath):
                os.remove(self.best_filepath)
            super()._save_checkpoint(trainer, filepath)

    checkpoint_callback = OverwriteModelCheckpoint(
        dirpath="./",  # 保存路径
        filename="best_model",     # 文件名格式（包含epoch和验证损失）
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_weights_only=False,   # False=保存完整模型（含优化器、参数）；True=仅保存权重
    )

    model = NeRFModel(opt)
    datamodule = hydra.utils.instantiate(opt.dataset)

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[checkpoint_callback],  # 传入checkpoint回调
        max_epochs=opt.trainer_args.max_epochs,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint('model.ckpt')

    # model = NeRFModel(opt)
    # datamodule = hydra.utils.instantiate(opt.dataset)
    # trainer = pl.Trainer(accelerator='gpu',
    #                      **opt.trainer_args)
    #
    # trainer.fit(model, datamodule=datamodule)
    # trainer.save_checkpoint('best_model.ckpt')


if __name__ == "__main__":
    main()

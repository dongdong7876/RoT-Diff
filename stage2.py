"""
Stage2: prior learning

run `python stage2.py`
"""
import os
import glob
from argparse import ArgumentParser
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from experiments.exp_stage2 import ExpStage2
from utils import load_yaml_param_settings
from data_factory.data_loader_contamination import get_loader_segment

def load_args():
    parser = ArgumentParser(description="Stage 2: Regime-Adaptive Conditional Generation")
    parser.add_argument('--data_name', type=str, required=True, help="Dataset name (e.g., SMD, SWaT, PSM, MSL, WADI)")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int)
    parser.add_argument('--data_path', type=str, default=None, help="Path to dataset")

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join('dataset', args.data_name)
    if args.config is None:
        args.config = os.path.join('config', f'config_{args.data_name}.yaml')

    return args


def train_stage2(config: dict,
                 data_name: str,
                 in_channels: int,
                 train_data_loader: DataLoader,
                 val_data_loader: DataLoader,
                 gpu_device_ind: list,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'RoT-Diff-stage2'
    wandb_logger = WandbLogger(project=project_name, name=data_name, config=config)
    # 1. Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['trainer_params']['save_dir'], data_name),
        filename='stage2-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 2. Trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer_params']['max_epochs']['stage2'],
        accelerator='gpu',
        devices=gpu_device_ind,
        strategy='ddp' if len(gpu_device_ind) > 1 else 'auto',
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        gradient_clip_val=config['trainer_params']['grad_clip_norm'],
        check_val_every_n_epoch=1,
    )

    # 3. Model
    stage2 = ExpStage2(in_channels=in_channels, data_name=data_name, config=config)

    # 4. Fit (Training)
    print("======================TRAIN MODE======================")
    trainer.fit(stage2, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    in_channels = config['dataset']['in_channels']
    batch_size = config['exp_params']['batch_sizes']['stage2']
    win_size = config['dataset']["win_size"]
    num_workers = config['dataset']["num_workers"]
    train_split = config['dataset']["train_split"]

    loaders = [get_loader_segment(data_path=args.data_path, batch_size=batch_size,
                                  win_size=win_size,
                                  train_split=train_split,
                                  mode=mode,
                                  num_workers=num_workers,
                                  data_name=args.data_name)
               for mode in ['train', 'val']]

    train_data_loader, val_data_loader = loaders
    # train
    train_stage2(config, data_name=args.data_name, in_channels=in_channels,
                 train_data_loader=train_data_loader,
                 val_data_loader=val_data_loader, gpu_device_ind=args.gpu_device_ind)

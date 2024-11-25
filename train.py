import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from argparse import ArgumentParser

from model.lightning_module import FasterRCNNModule
from data.coco_module import CocoDataModule

def main(args):
    # Initialize logger
    wandb_logger = WandbLogger(
        project="faster-rcnn-efficientnet",
        name=args.experiment_name,
        log_model=True
    )

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize data module
    data_module = CocoDataModule(
        train_ann_file=args.train_ann_file,
        train_img_dir=args.train_img_dir,
        val_ann_file=args.val_ann_file,
        val_img_dir=args.val_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize model
    model = FasterRCNNModule(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        pretrained=args.pretrained,
        val_ann_file=args.val_ann_file  # Pass validation annotation file
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.num_gpus > 1 else "auto",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        precision=16 if args.use_amp else 32,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument('--train_ann_file', type=str, required=True)
    parser.add_argument('--train_img_dir', type=str, required=True)
    parser.add_argument('--val_ann_file', type=str, required=True)
    parser.add_argument('--val_img_dir', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=91)  # COCO has 90 classes + background

    # Training arguments
    parser.add_argument('--experiment_name', type=str, default='faster-rcnn-efficientnet')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    args = parser.parse_args()
    main(args)


import torch
import json
import os
import pytorch_lightning as pl
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config, create_demo_callback_from_config
from downstream_tasks.eff_models.lora_dit_wrapper import LoRADiTWrapper
from downstream_tasks.configs.lora_config import get_lora_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    args = get_all_args()
    seed = args.seed
    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    
    pl.seed_everything(seed, workers=True)

    # Create training data loader
    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )
    
    # Create validation data loader
    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )
    
    # Create base model
    base_model = create_model_from_config(model_config)
    
    # Load pretrained weights if needed
    if args.pretrained_ckpt_path:
        copy_state_dict(base_model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    
    # Apply LoRA
    lora_config = get_lora_config()
    lora_model = LoRADiTWrapper(base_model, lora_config)
    
    # Create training wrapper (use existing DiffusionCondTrainingWrapper)
    training_wrapper = create_training_wrapper_from_config(model_config, lora_model)
    
    # Setup callbacks
    exc_callback = ExceptionCallback()
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    # Setup logger

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name)
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints") 
        else:
            checkpoint_dir = None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None


    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    
    # Combine configs for logging
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})
    args_dict.update({"lora_config": lora_config})
    
    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    
    # Setup trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        precision=args.precision,
        max_epochs=100,
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        val_check_interval=500,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accum_batches,
    )
    
    # Train with both dataloaders
    trainer.fit(training_wrapper, train_dl, val_dl)

if __name__ == "__main__":
    main()

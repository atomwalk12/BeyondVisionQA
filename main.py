from config import dataset_config, model_config
from qadataset import QADataset
from model import get_model
from model import PushToHubCallback
from trainer_blip2 import BLIP2PLModule
from trainer_instructblip import InstructBLIPSqaPLModule
import lightning as L
from config import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger



def train(module: L.LightningModule):
    generate_parameters = model_config['generate_parameters']
    hyperparams = model_config['hyperparameters']

    early_stop_callback = EarlyStopping(monitor="wup_measure", patience=3, verbose=False, mode="min")


    wandb_logger = WandbLogger(project=wandb['project'], name=wandb['name'])

    wandb_logger.experiment.config.update(hyperparams)
    wandb_logger.experiment.config.update(generate_parameters)

    trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=hyperparams.get("max_epochs"),
            accumulate_grad_batches=hyperparams.get("accumulate_grad_batches"),
            check_val_every_n_epoch=hyperparams.get("check_val_every_n_epoch"),
            gradient_clip_val=hyperparams.get("gradient_clip_val"),
            precision="16-mixed",
            num_sanity_val_steps=0,
            default_root_dir=model_config['local_checkpoint_dir'],
            logger=wandb_logger,
            callbacks=[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(module)



if __name__ == '__main__':
    # import ntlk
    # ntlk.download('wordnet')
    # torch.multiprocessing.set_start_method('spawn')
    train_dataset = QADataset(dataset_config, split="train[:100]")
    val_dataset = QADataset(dataset_config, split="train[:20]")
    
    model = get_model(quantization='8bit')
    
    hyperparameters = model_config['hyperparameters']
    if model_config['target'] == 'blip2':
        module = BLIP2PLModule(hyperparameters, model, train_dataset, val_dataset)
    elif model_config['target'] == 'instructblip':
        module = InstructBLIPSqaPLModule(hyperparameters, model, train_dataset, val_dataset)
    train(module)

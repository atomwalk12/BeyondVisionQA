import math
import lightning as L
import torch
from config import dataset_config, model_config, wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from lightning.pytorch.loggers import WandbLogger
from config import metrics
from transformers import AutoProcessor

generate_parameters = model_config['generate_parameters']
MAX_LENGTH = generate_parameters['max_new_tokens']
PEFT_ID = model_config['peft_id']
MODEL_ID = model_config['model_id']
WANDB_PROJECT = wandb['project']
WANDB_NAME = wandb['name']

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

if dataset_config['name'] == "scienceqa":
    from dataset_configs.scienceqa import translate
    
if dataset_config['name'] == "daquar":
    from dataset_configs.dquar import translate    

if dataset_config['name'] == "easy-vqa":
    from dataset_configs.easy_vqa import translate   


class InstructBLIPModelPLModule(L.LightningModule):
    def __init__(self, hyperparameters, model, train_dataset, val_dataset):
        super().__init__()
        self.hyperparams = hyperparameters
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = hyperparameters.get("batch_size")
        


    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            qformer_input_ids=qformer_input_ids,
                            qformer_attention_mask=qformer_attention_mask,
                            pixel_values=pixel_values,
                            labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        inputs, answers = batch

        # auto-regressively generate token IDs

        
        generated_ids = self.model.generate(**inputs,
                                            **generate_parameters)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print(f"==================Dataset batch: {batch_idx}/{self.val_dataset.dataset_length // self.batch_size}==================")
        scores = []
        i = 0
        for pred, answer in zip(predictions, answers):
            print(f"Question: {self.val_dataset.dataset[batch_idx*self.batch_size+i]['question']}")
            print(f"Prediction: {pred}")
            print(f"Answer: {answer}")
            i += 1

        for metric in metrics:
            scores = metric.compute(predictions=predictions, references=answers, model=self)
            
        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        #return optimizer
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hyperparams.get("lr"),
            betas=self.hyperparams['betas'],
            weight_decay=self.hyperparams['weight_decay']
        )

        # Calculate total steps based on epochs and batch size
        total_epochs = self.hyperparams['max_epochs']
        dataset_size = len(self.train_dataset)  # Assuming you have access to the dataset size
        batch_size = self.hyperparams['batch_size']
        steps_per_epoch = dataset_size // batch_size  # Integer division
        total_steps = total_epochs * steps_per_epoch

        # Convert warmup steps to warmup epochs if necessary
        warmup_epochs = self.hyperparams['warmup_epochs']
        warmup_steps = math.ceil(warmup_epochs * steps_per_epoch)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
        return [optimizer], [scheduler]

    
class InstructBLIPSqaPLModule(InstructBLIPModelPLModule):
    
    def __init__(self, config, model, train_dataset, val_dataset):
        super().__init__(config, model, train_dataset, val_dataset)
    
    def train_collate_fn(examples):
        images = []
        texts = []
        for example in examples:
            image, ground_truth = example
            input = translate(ground_truth, training=True)
            
            images.append(image)
            texts.append(input)    

        inputs = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        qformer_input_ids = inputs['qformer_input_ids']
        qformer_attention_mask = inputs['qformer_attention_mask']
        pixel_values = inputs['pixel_values']#

        return input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values, labels


    def eval_collate_fn(examples):
        images = []
        texts = []
        answers = []
        for example in examples:
            image, ground_truth = example
            input, output = translate(ground_truth, training=False)
            
            images.append(image)
            texts.append(input) 
            answers.append(output)

        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        return inputs, answers
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=InstructBLIPSqaPLModule.train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=InstructBLIPSqaPLModule.eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)
    

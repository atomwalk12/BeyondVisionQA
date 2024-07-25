from qadataset import QADataset
from config import dataset_config, model_config

train_dataset = QADataset(dataset_config, split="train[:6]")
val_dataset = QADataset(dataset_config, split="train[:6]")

# from model import get_model

# model = get_model(quantization='8bit')
from torch.utils.data import Dataset, DataLoader
from trainer_blip2 import BLIP2ModelPLModule
from trainer_blip2 import BLIP2PLModule
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)
# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


train_dataloader = DataLoader(train_dataset, collate_fn=BLIP2PLModule.train_textual_labels, batch_size=8, shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

from transformers import TrainingArguments, Trainer, AutoTokenizer
from dataset_configs.easy_vqa import translate
from dataset_configs.easy_vqa import filter, get_image
from PIL import Image

training_args = TrainingArguments(output_dir="blip2-easyvqa-hf_trainer", eval_strategy="epoch", )
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

train_dataset[0]
def tokenize_function(examples):
    texts = []
    images = []
    for i in range(len(examples['question'])):  # Iterate over the indices
        item = {
            'question': examples['question'][i],
            'answer': examples['answer'][i],
            'id': examples['id'][i],
            'label': examples['label'][i],
            'path': examples['path'][i]
        }
        texts.append(translate(item, training=True))
        images.append(Image.open(examples['path'][i]))
    return processor(images=images, text=texts, padding=True, return_tensors="pt")

tokenized_datasets = train_dataset.dataset.map(tokenize_function, batched=True)
tokenized_datasets = val_dataset.dataset.map(tokenize_function, batched=True)

from model import BertScoreMetric
import numpy as np

metric = BertScoreMetric()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, lang='en')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
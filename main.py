from config import DATASET_NAME
from qadataset import QADataset
from trainer import BLIPSqaPLModule, train
from config import hyperparameters
from model import get_model


if __name__ == '__main__':
  # torch.multiprocessing.set_start_method('spawn')
    train_dataset = QADataset(DATASET_NAME,  split="train[:5]")
    val_dataset = QADataset(DATASET_NAME, split="validation[:10]")
    
    model = get_model(quantization='4bit')
    
    module = BLIPSqaPLModule(hyperparameters, model, train_dataset, val_dataset)
    train(module)
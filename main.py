from config import dataset_config
from qadataset import QADataset
from trainer import BLIPSqaPLModule, train
from config import hyperparameters
from model import get_model


if __name__ == '__main__':
  # import ntlk
  # ntlk.download('wordnet')

  # torch.multiprocessing.set_start_method('spawn')
  config = dataset_config['config']
  train_dataset = QADataset(config, split="train")
  val_dataset = QADataset(config, split="eval")
  
  # train_dataset = QADataset(DATASET_NAME,  split="train[:5]")
  # val_dataset = QADataset(DATASET_NAME, split="validation[:10]")
  
  model = get_model(quantization='4bit')
  
  module = BLIPSqaPLModule(hyperparameters, model, train_dataset, val_dataset)
  train(module)

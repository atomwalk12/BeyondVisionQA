from config import dataset_config, model_config
from qadataset import QADataset
from trainer import BLIPSqaPLModule, train
from model import get_model


if __name__ == '__main__':
  # import ntlk
  # ntlk.download('wordnet')

  # torch.multiprocessing.set_start_method('spawn')
  train_dataset = QADataset(dataset_config, split="train")
  val_dataset = QADataset(dataset_config, split="val")
  
  model = get_model(quantization='4bit')
  
  hyperparameters = model_config['hyperparameters']
  module = BLIPSqaPLModule(hyperparameters, model, train_dataset, val_dataset)
  train(module)

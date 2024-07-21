


from typing import Dict

from config import dataset_config
from datasets import load_dataset
from torch.utils.data import Dataset

if dataset_config.get('dataset_name') == "scienceqa":
    from dataset_configs.scienceqa import filter, get_image

class QADataset(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train"
    ):
        super().__init__()

        self.split = split
        self.dataset_name = dataset_name_or_path

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset = filter(self.dataset)
        self.dataset_length = len(self.dataset)

        self.token_sequences = []
        for sample in self.dataset:
            ground_truth = sample
            self.token_sequences.append(ground_truth)


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        image = get_image(self.dataset, index)
        target_sequence = self.token_sequences[index]

        return image, target_sequence

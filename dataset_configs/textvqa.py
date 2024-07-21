from collections import defaultdict

def filter(dataset):
    return dataset

def get_image(dataset, index):
    sample = dataset[index]
    return sample["image"]


def translate(example, training, options=["A", "B", "C", "D", "E"]):
    pass
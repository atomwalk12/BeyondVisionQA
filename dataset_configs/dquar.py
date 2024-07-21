import os
from datasets import load_dataset
from PIL import Image



def translate(example, training):
    
    input = example["question"]
    output = example["answer"]
    
    if training:
        prompt = f"{input}. Answer: {output}."
        return prompt
    else:
        prompt = f"{input}. Answer:"
        return prompt, output

def load_data(split:str):
    if '[' in split:
        split, count= split.split("[:")
        count = count[:-1]
    else:
        count = None
    dataset = load_dataset(
        "csv", 
        data_files={
            f"{split}": os.path.join("data", "daquar", f"data_{split}.csv")
        }
    )

    with open(os.path.join("data", "daquar", "answer_space.txt")) as f:
        answer_space = f.read().splitlines()

    dataset = dataset.map(
        lambda examples: {
            'label': [
                answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                for ans in examples['answer']
            ]
        },
        batched=True
    )
    if count is not None:
        return dataset[split].select(range(int(count)))
    else:
        return dataset[split]


def filter(dataset):
    return dataset

def get_image(data, id):
    image = Image.open(os.path.join("data", "daquar", "images", data[id]["image_id"] + ".png"))
    return image

def get_answer_space():
    with open(os.path.join("data", "daquar", "answer_space.txt")) as f:
        answer_space = f.read().splitlines()
        return answer_space
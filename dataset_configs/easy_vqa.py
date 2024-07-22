from easy_vqa import get_train_questions, get_test_questions, get_answers, get_train_image_paths, get_test_image_paths
import os
from datasets import load_dataset, Dataset
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
        
    if split == 'train' or split == 'val':
        dataset = get_train_questions()
        paths = get_train_image_paths()
    elif split == 'test':
        dataset = get_test_questions()
        paths = get_test_image_paths()

    dataset = { 'question': dataset[0], 'answer': dataset[1], 'id': dataset[2]}
    dataset = Dataset.from_dict(dataset)

    answer_space = get_answers()
    

    dataset = dataset.map(
        lambda examples: {
            'label': [
                answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                for ans in examples['answer']
            ],
            'path': [
                paths[ans] # Select the 1st answer if multiple answers are provided
                for ans in examples['id']
            ],
        },
        batched=True
    )
    
    if split == 'train' or split == 'val':
        split = 'train' if split == 'train' else 'test'
        # Shuffling is performed at a later stage.
        dataset = { split: dataset.train_test_split(
                test_size=0.2, 
                shuffle=False
            )[split]
        }[split]
    
    if count is not None:
        return dataset.select(range(int(count)))
    else:
        return dataset


def filter(dataset):
    return dataset

def get_image(dataset, id):
    image = Image.open(dataset[id]['path'])
    return image



MAX_LENGTH = 384
PEFT_ID = "atomwalk12/instructblip-aw12-sqa-v2"
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
WANDB_PROJECT = "InstructBLIP"
WANDB_NAME = PEFT_ID

CONTINUE = True
REVISION = "1e6ce1964275c7de184b6e2817bacac970dbafe2"

dataset_config = {
    'config':{ 
        # 'path': "cambridgeltl/vsr_random",: https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data
        # 'facebook/textvqa', # Alternatives: derek-thomas/ScienceQA
        # 'data_files': {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        # local: DAQUAR
        'path': 'local'
    },
    'dataset_name': 'daquar', # Alternatives: textvqa, scienceqa, vsr_random
    'checkpoint_dir': "./InstructBLIP"
}

hyperparameters = {
    "target_model": "blip",
    "max_epochs": 20,
    "warmup_epochs": 1.67,
  # "val_check_interval": 0.2,
    "check_val_every_n_epoch": 2,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "lr": 1e-4,
    "batch_size": 7,
    "seed":1337,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./result",
    "verbose": True,
    "betas": (0.9, 0.999),
    "weight_decay": 0.05
}




if hyperparameters['target_model'] == 'blip':
    from config import MAX_LENGTH

    generate_parameters = {
        "do_sample": True,
        "num_beams": 5,
        "max_new_tokens": MAX_LENGTH,
        "min_length": 1,
        "top_p": 0.9,
        "repetition_penalty": 1.5,
        "length_penalty": 1.0,
        "temperature": 1,
    }
    

    lora_blip_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj','fc1', 'fc2']
    }

if dataset_config['dataset_name'] == 'scienceqa':
    from model import EditDistanceMetric, BertScoreMetric, RougeMetric
    
    metrics = [ EditDistanceMetric(), BertScoreMetric("bertscore"), RougeMetric("rouge") ]

if dataset_config['dataset_name'] == 'textvqa':
    from config import MAX_LENGTH
    from model import VQA

    metrics = [ EditDistanceMetric(), BertScoreMetric("bertscore"), RougeMetric("rouge") ]

if dataset_config['dataset_name'] == 'daquar':
    from dataset_configs.dquar import load_data, get_answer_space
    from model import WUPMeasure, F1ScoreMetric, AccuracyMetric
    
    dataset_config['config'].update({'load_fn': load_data})
    
    answer_space = get_answer_space()
    
    metrics = [ AccuracyMetric(), F1ScoreMetric(), WUPMeasure(answer_space) ]

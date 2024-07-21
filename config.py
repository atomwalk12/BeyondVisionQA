MAX_LENGTH = 384
PEFT_ID = "atomwalk12/instructblip-aw12-sqa-v2"
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
WANDB_PROJECT = "InstructBLIP"
WANDB_NAME = PEFT_ID
DATASET_NAME = "derek-thomas/ScienceQA"
CONTINUE = False
REVISION = None

dataset_config = {
    'dataset_name': 'scienceqa',
    'checkpoint_dir': "./InstructBLIP"
}

hyperparameters = {
    "max_epochs": 20,
    "warmup_epochs": 1.67,
  # "val_check_interval": 0.2,
    "check_val_every_n_epoch": 2,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "lr": 1e-4,
    "batch_size": 2,
    "seed":1337,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./result",
    "verbose": True,
    "betas": (0.9, 0.999),
    "weight_decay": 0.05
}

if dataset_config['dataset_name'] == 'scienceqa':
    from config import MAX_LENGTH
    from datasets import load_metric

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

    bertscore = load_metric("bertscore")
    rouge = load_metric("rouge")
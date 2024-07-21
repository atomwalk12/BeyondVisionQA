from transformers import BitsAndBytesConfig, InstructBlipForConditionalGeneration
from peft import PeftConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from config import PEFT_ID, REVISION, CONTINUE, MODEL_ID


def get_model(quantization='4bit'):
    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    
    bb_config_4b = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    ##########################################

    ################## 8bit ##################
    bb_config_8b = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    ##########################################

    def quantization_config(quantization):
        if quantization == "8bit":
            return bb_config_8b
        else:
            return bb_config_4b
        
    
    config = None
    if CONTINUE:
        config = PeftConfig.from_pretrained(PEFT_ID, revision=REVISION)
        config.inference_mode = False
        print("Using checkpoint")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            quantization_config=quantization_config(quantization)
        )
    elif quantization == "none":
        print("Not using checkpoint")
        model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")
    else: 
        print("Not using checkpoint")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=quantization_config(quantization)
        )
        

    # Let's define the LoraConfig
    # Configurable layers for BLIP:
    # 'down_proj', 'key', 'v_proj', 'query', 'q_proj', 'fc1', 'up_proj', 'lm_head', 'dense', 'fc2', 
    # 'k_proj', 'o_proj', 'projection', 'value', 'gate_proj', 'qkv', 'language_projection'
    # Another possible configuration:
    # Configuration that loaded ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'up_proj'] # Does not have fc1 fc2
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj','fc1', 'fc2']
    )

    if not CONTINUE:
        print("Not using checkpoint")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        print("Using checkpoint")
        model = prepare_model_for_kbit_training(model)
        # model = PeftModel.from_pretrained(model, PEFT_ID, is_trainable=True)
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model
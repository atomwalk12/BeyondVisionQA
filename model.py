from abc import abstractmethod
from transformers import BitsAndBytesConfig, InstructBlipForConditionalGeneration
from peft import PeftConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_metric
import lightning as L
import numpy as np
from nltk import edit_distance
import re
from sklearn.metrics import accuracy_score, f1_score

from config import model_config
from nltk.corpus import wordnet


PEFT_ID = model_config['peft_id']
REVISION = model_config['revision']
CONTINUE = model_config['continue']
MODEL_ID = model_config['model_id']
hyperparameters = model_config['hyperparameters']


def get_model(quantization='4bit'):
    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # - Continue from huggingface url using QLora
    # - Standard Lora
    # - QLora
    
    
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
        
    if CONTINUE:
        print(f"Using checkpoint: {PEFT_ID}/{REVISION}")
        
        config = PeftConfig.from_pretrained(PEFT_ID, revision=REVISION)
        config.inference_mode = False
       
        model = InstructBlipForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            quantization_config=quantization_config(quantization)
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
    else:
        if quantization == "none":
            print("Loading model without using quantization.")
            model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")

        else: 
            print(f"Loading model with quantization ({quantization})")
            model = InstructBlipForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                quantization_config=quantization_config(quantization)
            )
        
        if hyperparameters['target_model'] == 'blip':
            from config import model_config
            lora_config = model_config['peft']
        else:
            raise KeyError(f"Undefined lora config for {hyperparameters['target_model']}")
        
        lora_config = LoraConfig(**lora_config)
            
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    return model


class Metric():
    
    @abstractmethod
    def compute(self, predictions,references):
        pass

class RougeMetric():
    
    def __init__(self, metric) -> None:
        self.metric = load_metric("rouge")
    
    def compute(self, predictions,references, model: L.LightningModule):
        scores = self.metric.compute(predictions=predictions, references=references)
        model.log("val_rouge_f1", scores["rouge1"].mid.fmeasure)
        print("val_rouge_f1:", scores["rouge1"].mid.fmeasure)
        return scores
    
class BertScoreMetric():
    
    def __init__(self, metric) -> None:
        self.metric = load_metric("bertscore")
    
    def compute(self, predictions,references, model: L.LightningModule):
        scores = self.metric.compute(predictions=predictions, references=references)
        model.log("val_bertscore_f1", np.mean(scores["f1"])) 
        print("val_bertscore_f1:", np.mean(scores["f1"]), "\n\n")
        return scores


    
class AccuracyMetric():
    
    def compute(self, predictions, references, model: L.LightningModule):
        scores = accuracy_score(y_pred=predictions, y_true=references)
        model.log("accuracy", scores) 
        print("accuracy:", scores)
        return scores


class F1ScoreMetric():
    
    def compute(self, predictions, references, model: L.LightningModule):
        scores = f1_score(y_pred=predictions, y_true=references, average='macro')
        model.log("f1score", scores) 
        print("f1score:", scores)
        return scores

class WUPMeasure():
    def __init__(self, answer_space = None) -> None:
        self.answer_space = answer_space
    
    def compute(self, predictions, references, model: L.LightningModule):
        scores = self.batch_wup_measure(predictions=predictions, references=references)
        model.log("wup_measure", scores) 
        print("WUP Measure:", scores, "\n\n")
        return scores
    
    def wup_measure(self, a, b,similarity_threshold=0.925):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        def get_semantic_field(a):
            weight = 1.0
            semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
            return (semantic_field,weight)


        def get_stem_word(a):
            """
            Sometimes answer has form word\d+:word id.
            If so we return word and down-weight
            """
            weight = 1.0
            return (a,weight)


        global_weight=1.0

        (a,global_weight_a)=get_stem_word(a)
        (b,global_weight_b)=get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0


        interp_a,weight_a = get_semantic_field(a) 
        interp_b,weight_b = get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we down weight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score 
    
    def batch_wup_measure(self, references, predictions):
        wup_scores = [self.wup_measure(label, pred) for label, pred in zip(references, predictions)]
        return np.mean(wup_scores)

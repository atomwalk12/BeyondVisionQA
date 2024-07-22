
from utils import load_config

# Load the configuration
dataset_config, model_config, wandb = load_config('experiment2.yaml')

if dataset_config['name'] == 'scienceqa':
    from model import EditDistanceMetric, BertScoreMetric, RougeMetric
    
    metrics = [ EditDistanceMetric(), BertScoreMetric("bertscore"), RougeMetric("rouge") ]

if dataset_config['name'] == 'textvqa':
    from config import MAX_LENGTH
    from model import VQA

    metrics = [ EditDistanceMetric(), BertScoreMetric("bertscore"), RougeMetric("rouge") ]

if dataset_config['name'] == 'daquar':
    from dataset_configs.dquar import load_data, get_answer_space
    from model import WUPMeasure, F1ScoreMetric, AccuracyMetric
    
    dataset_config.update({'load_fn': load_data}) 
    answer_space = get_answer_space()
    
    metrics = [ AccuracyMetric(), F1ScoreMetric(), WUPMeasure(answer_space) ]


if dataset_config['name'] == 'easy-vqa':
    from dataset_configs.easy_vqa import load_data
    from model import WUPMeasure, F1ScoreMetric, AccuracyMetric
    from easy_vqa import get_answers
    
    dataset_config.update({'load_fn': load_data}) 
    answer_space = get_answers()
    
    metrics = [ AccuracyMetric(), F1ScoreMetric(), WUPMeasure(answer_space) ]

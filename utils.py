import yaml
    
def load_config(experiment_id):
    with open(f"experiments/{experiment_id}", 'r') as file:
        config = yaml.safe_load(file)
    return config.values()



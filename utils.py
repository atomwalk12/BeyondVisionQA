import yaml
    
def load_config(experiment_id):
    
    def none_constructor(loader, node):
        return None

    # Register the custom constructor for empty strings
    yaml.add_constructor('tag:yaml.org,2002:null', none_constructor)
    with open(f"experiments/{experiment_id}", 'r') as file:
        config = yaml.safe_load(file)
    return config.values()



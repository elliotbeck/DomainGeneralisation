from models.simple_nn import basic_nn

def get_model(name, config):
    basic_nn(config.num_classes, config)

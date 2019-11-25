from models.DAOT_mnist import basic_nn
from models.DAOT_mnist import generator
from models.DAOT_mnist import critic

def get_model(name, config):
    if name == "basic_nn":
        return basic_nn(config.num_classes, config)
    elif name == "critic":
        return critic(config.num_classes, config.resnet_weights, config)
    elif name == "generator":
        return generator(config)
    else:
        raise NotImplementedError
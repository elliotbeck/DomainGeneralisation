from models.densenet import DenseNet
from models.DAOT import ResNet50
from models.DAOT import generator
from models.DAOT import critic

def get_model(name, config):
    if name == "densenet121":
        return DenseNet(config.num_classes, config.densenet_weights, config)
    elif name == "ResNet50":
        return ResNet50(config.num_classes, config.resnet_weights, config)
    elif name == "critic":
        return critic(config.num_classes, config.resnet_weights, config)
    elif name == "generator":
        return generator(config)
    else:
        raise NotImplementedError
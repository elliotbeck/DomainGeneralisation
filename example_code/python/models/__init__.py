from models.densenet import DenseNet

def get_model(name, config):
    if name == "densenet121":
        return DenseNet(config.num_classes, config.densenet_weights, config)
    else:
        raise NotImplementedError
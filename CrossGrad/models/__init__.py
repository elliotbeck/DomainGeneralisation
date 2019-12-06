from models.CrossGrad import model_domain
from models.CrossGrad import model_label

def get_model(name, config):
    if name == "classifier_domain":
        return model_domain(config.num_classes_domain, config.resnet_weights, config)
    elif name == "classifier_label":
        return model_label(config.num_classes_label, config.resnet_weights, config)
    else:
        raise NotImplementedError
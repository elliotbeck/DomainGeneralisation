from models.MetaReg import model_feature
from models.MetaReg import model_task
from models.MetaReg import model_regularizer

def get_model(name, config, model_feature=model_feature):
    if name == "feature_network":
        return model_feature(config.resnet_weights, config)
    elif name == "classifier_task":
        return model_task(config.num_classes_label, config, model_feature)
    elif name == "regularizer_network":
        return model_regularizer(config)
    else:
        raise NotImplementedError
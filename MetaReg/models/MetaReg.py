import tensorflow as tf


class model_feature(tf.keras.Model):
    INPUT_SHAPE = [227, 227]

    def __init__(self, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights= resnet_weights, input_shape=in_shape)) 
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Flatten())                                  
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return model_feature.INPUT_SHAPE

class model_task(tf.keras.Model):
    INPUT_SHAPE = [227, 227]
    def __init__(self, num_classes_labels, config, model_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.model = tf.keras.models.Sequential()
        self.model.add(model_feature)
        self.model.add(tf.keras.layers.Dense(num_classes_labels, activation='softmax', use_bias=False))
        self.model.build([None] + self.input_shape + [3])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return model_task.INPUT_SHAPE

class model_regularizer(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.model = tf.keras.models.Sequential()
        #self.model.add(tf.keras.layers.Flatten(input_shape=(2048,7))) 
        self.model.add(tf.keras.layers.Dense(1, activation = 'linear', 
                        input_shape = (14336,), use_bias=False))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)


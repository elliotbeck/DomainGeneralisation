import tensorflow as tf


class model_domain(tf.keras.Model):
    INPUT_SHAPE = [227, 227]

    def __init__(self, num_classes_domain, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights= resnet_weights, input_shape=in_shape))
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())                                    
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(1048, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(num_classes_domain))
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return model_domain.INPUT_SHAPE


class model_label(tf.keras.Model):
    INPUT_SHAPE = [227, 227]

    def __init__(self, num_classes_label, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights= resnet_weights, input_shape=in_shape))    
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())                                    
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(2048, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(1028, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(num_classes_label))
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return model_label.INPUT_SHAPE
        
import tensorflow as tf
import util


class ResNet50(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, num_classes, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.Sequential([
            tf.compat.v1.keras.applications.ResNet50(include_top=False,
                                                        weights=resnet_weights, input_shape=in_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dense(34, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            #tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return ResNet50.INPUT_SHAPE


class generator(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=(1), filters=3 ,strides=(1), input_shape=in_shape, padding="same",
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001), 
                                        activation='relu'),
            tf.keras.layers.Conv2D(kernel_size=(1), filters=3,strides=(1), padding="same", 
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001), activation="tanh")
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        X_shortcut = inputs
        return tf.math.add(self.model(inputs, training, mask), X_shortcut) # have to replace 1 with lambda from config

    @property
    def input_shape(self):
        return generator.INPUT_SHAPE

class critic(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, num_classes, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.Sequential([
            tf.compat.v1.keras.applications.ResNet50(include_top=False,
                                                        weights=resnet_weights, input_shape=in_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
            #tf.keras.layers.Dropout(0.5),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dense(34, activation='relu')
            #tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return critic.INPUT_SHAPE
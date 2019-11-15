import tensorflow as tf
import util
import json

with open('DAOT/configs/config_class_daot.json', 'r') as myfile:
    data=myfile.read()
config_dic2 = json.loads(data)
config_seed = config_dic2["seed"]


class basic_nn(tf.keras.Model):
    INPUT_SHAPE = [14, 14]

    def __init__(self, num_classes, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = [2] + self.input_shape 

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None] + [2] + self.input_shape)  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return basic_nn.INPUT_SHAPE


class generator(tf.keras.Model):
    INPUT_SHAPE = [14, 14]

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = [2] + self.input_shape

        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(kernel_size=(1), filters=14 ,strides=(1), 
                                    input_shape=in_shape, padding="same",
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(), 
                                    activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(kernel_size=(1), filters=14,strides=(1), padding="same", 
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(), 
                                    activation='tanh')
        ])
        self.model.build([None] + [2] + self.input_shape)  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        X_shortcut = inputs
        print(self.model(inputs, training, mask))
        output = tf.keras.layers.add([config_dic2["lambd"]*self.model(inputs, training, mask), X_shortcut])
        #output = tf.keras.activations.tanh(output)
        return output
    
        #return tf.math.add(self.model(inputs, training, mask), X_shortcut) # have to replace 1 with lambda from config

    @property
    def input_shape(self):
        return generator.INPUT_SHAPE

class critic(tf.keras.Model):
    INPUT_SHAPE = [14, 14]

    def __init__(self, num_classes, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = [2] + self.input_shape 

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
        ])
        self.model.build([None] + [2] + self.input_shape)  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return critic.INPUT_SHAPE
import numpy as np 
import tensorflow as tf
import h5py
import os
import argparse
import random


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# TODO shuffle the datasets before every epoch
# such that all samples get used in training


parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--hidden_dim', type=int, default=1028)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.0001)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--shuffle_buffer_size', type=int, default=2000)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--seed', type=int, default=1)
flags = parser.parse_args()

random.seed(flags.seed)

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

# read in data

hf = h5py.File('/cluster/work/math/ebeck/data/pacs/sketch_train.hdf5', 'r')
x_train1 = np.array(hf["images"][:])
y_train1 = np.array(hf["labels"][:])-1


hf = h5py.File('//cluster/work/math/ebeck/data/pacs/art_painting_train.hdf5', 'r')
x_train2 = np.array(hf["images"][:])
y_train2 = np.array(hf["labels"][:])-1


hf = h5py.File('/cluster/work/math/ebeck/data/pacs/photo_train.hdf5', 'r')
x_train3 = np.array(hf["images"][:])
y_train3 = np.array(hf["labels"][:])-1


hf = h5py.File('/cluster/work/math/ebeck/data/pacs/cartoon_test.hdf5', 'r')
x_test = np.array(hf["images"][:])
y_test = np.array(hf["labels"][:])-1


# create tf datasets
train_ds1 = tf.data.Dataset.from_tensor_slices((tf.cast(x_train1, dtype=tf.float32), y_train1))
train_ds1 = train_ds1.shuffle(flags.shuffle_buffer_size)
train_ds1 = train_ds1.batch(flags.batch_size)

train_ds2 = tf.data.Dataset.from_tensor_slices((tf.cast(x_train2, dtype=tf.float32), y_train2))
train_ds2 = train_ds2.shuffle(flags.shuffle_buffer_size)
train_ds2 = train_ds2.batch(flags.batch_size)

train_ds3 = tf.data.Dataset.from_tensor_slices((tf.cast(x_train3, dtype=tf.float32), y_train3))
train_ds3 = train_ds3.shuffle(flags.shuffle_buffer_size)
train_ds3 = train_ds3.batch(flags.batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((tf.cast(x_test, dtype=tf.float32), y_test))
test_ds = test_ds.shuffle(flags.shuffle_buffer_size)
test_ds = test_ds.batch(flags.batch_size)

# Build environments

envs = [
    train_ds1,
    train_ds2,
    train_ds3,
    test_ds
  ]

# build model

class ResNet50(tf.keras.Model):
    INPUT_SHAPE = [227, 227]

    def __init__(self, num_classes=7, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_shape = self.input_shape + [3]

        self.model = tf.keras.Sequential([
            tf.compat.v1.keras.applications.ResNet50(include_top=False,
                                                        weights='imagenet', input_shape=in_shape),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(34, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes)
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return ResNet50.INPUT_SHAPE

# initialize model

model = ResNet50()

# Define loss function helpers
# not possible to use tf.keras.losses.SparseCategoricalCrossentropy due to:
# https://github.com/tensorflow/tensorflow/issues/27875
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
def mean_nll(logits, y):
    return loss_object(tf.one_hot(tf.cast(y, dtype=tf.int32), axis=-1, depth = 7), logits)

def mean_accuracy(logits, y):
    accuracy = tf.math.reduce_mean(
        tf.where(tf.equal(y, tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int8)),
                    tf.ones_like(y, dtype=tf.float16),
                    tf.zeros_like(y, dtype=tf.float16)))
    return accuracy

def penalty(logits, y):
    with tf.GradientTape() as tape_src:
        scale = tf.ones(1,1)
        tape_src.watch(scale)
        loss = mean_nll(logits * scale, y)
        grad = tape_src.gradient(loss, scale)

    return tf.reduce_sum(grad**2)

# define optimizer

optimizer = tf.keras.optimizers.Adam(lr=flags.lr)

# define printing function

def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=2, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

    

# define train metrics

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.Mean(name='train_acc')
test_acc = tf.keras.metrics.Mean(name='test_acc')

# start loop

for step in range(flags.epochs):
    train_loss.reset_states()
    train_acc.reset_states()
    test_acc.reset_states()

    for env0, env1, env2, env3 in zip(envs[0], envs[1], envs[2], envs[3]):
        with tf.GradientTape() as tape_src:
            env = [[], [], [], []]
            env[0].append(mean_nll(model(env0[0]/255.), env0[1]))
            env[0].append(mean_accuracy(model(env0[0]/255.), env0[1]))
            env[0].append(penalty(model(env0[0]/255.), env0[1]))

            env[1].append(mean_nll(model(env1[0]/255.), env1[1]))
            env[1].append(mean_accuracy(model(env1[0]/255.), env1[1]))
            env[1].append(penalty(model(env1[0]/255.), env1[1]))

            env[2].append(mean_nll(model(env2[0]/255.), env2[1]))
            env[2].append(mean_accuracy(model(env2[0]/255.), env2[1]))
            env[2].append(penalty(model(env2[0]/255.), env2[1]))

            env[3].append(mean_nll(model(env3[0]/255.), env3[1]))
            env[3].append(mean_accuracy(model(env3[0]/255.), env3[1]))
            env[3].append(penalty(model(env3[0]/255.), env3[1]))

            train_nll = tf.reduce_mean([env[0][0], env[1][0], env[2][0]])
            train_accuracy = tf.reduce_mean([env[0][1], env[1][1], env[2][1]])
            train_penalty = tf.reduce_mean([env[0][2], env[1][2], env[2][2]])

            test_accuracy = env[3][1]

            train_loss(train_nll)
            train_acc(train_accuracy)
            test_acc(test_accuracy)

            tape_src.watch(train_nll)


            weight_norm = tf.zeros(1,1)
            for w in model.trainable_variables:
                weight_norm += tf.norm(w)**2

            loss = train_nll
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (flags.penalty_weight 
                if step >= flags.penalty_anneal_iters else 0.01)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            # update weights of classifier
            grads = tape_src.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    if step % 1 == 0:    
        pretty_print('epoch', 'train nll', 'train acc', 'test acc')
    
    pretty_print(
        np.int32(step+1),
        train_loss.result().numpy(),
        (train_acc.result()*100).numpy(),
        (test_acc.result()*100).numpy())
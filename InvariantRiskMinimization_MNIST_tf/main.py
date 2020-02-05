import numpy as np 
import tensorflow as tf
import h5py
import os
import argparse
import tensorflow_datasets as tfds
import local_settings
import util
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--epochs', type=int, default=501)
parser.add_argument('--batch_size', type=int, default=40000)
parser.add_argument('--shuffle_buffer_size', type=int, default=20000)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--seed', type=int, default=1)
flags = parser.parse_args()
random.seed(flags.seed)

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

# build model
class MLP(tf.keras.Model):
    INPUT_SHAPE = [14, 14]
    def __init__(self, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_shape = self.input_shape + [2]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(flags.hidden_dim, activation='relu', 
                                     kernel_initializer = tf.initializers.GlorotUniform()),
             tf.keras.layers.Dense(flags.hidden_dim, activation='relu', 
                                     kernel_initializer = tf.initializers.GlorotUniform()),
             tf.keras.layers.Dense(num_classes, activation ='softmax',
                                     kernel_initializer = tf.initializers.GlorotUniform())])
        self.model.build([None] + self.input_shape + [2])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)
    @property
    def input_shape(self):
        return MLP.INPUT_SHAPE

# initialize model
model = MLP()

# get datasets
def _preprocess_exampe(model, example, dataset_name, e):
    example["image"] = tf.cast(example["image"], dtype=tf.float64)/255.
    # 2x subsample for computational convenience
    example["image"] = example["image"][::2, ::2, :]
    example["image"] = tf.squeeze(example["image"], axis=-1)
    # Assign a binary label based on the digit; flip label with probability 0.25
    label = tf.cast([[example["label"] < 5]], dtype=tf.float32)
    label = util.tf_xor(label, util.tf_bernoulli(0.25, 1))
    # Assign a color based on the label; flip the color with probability e
    colors = util.tf_xor(label, util.tf_bernoulli(e, 1))
    re_colors = 1-colors
    re_colors = tf.cast(re_colors, dtype=tf.int32)
    # Apply the color to the image by zeroing out the other color channel
    if re_colors == tf.constant(0): 
        image = tf.stack([tf.zeros([14,14], dtype=tf.float64),
        example["image"]], axis=-1)
    else: 
        image = tf.stack([example["image"], tf.zeros([14,14], 
        dtype=tf.float64)], axis=-1)
    example["image"] = image
    example["label"] = tf.squeeze(label)
    return example

def _get_dataset(dataset_name, model, split, batch_size, e):
    dataset, _ = tfds.load(dataset_name, data_dir=local_settings.TF_DATASET_PATH, 
        split=split, with_info=True)
    dataset = dataset.map(lambda x: _preprocess_exampe(model, x, dataset_name, e))
    dataset = dataset.shuffle(flags.shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset

train_ds1 = _get_dataset('mnist', model,
        split=tfds.Split.TRAIN.subsplit(tfds.percent[:50]), 
        batch_size=flags.batch_size, e = 0.2)
train_ds2 = _get_dataset('mnist', model, 
        split=tfds.Split.TRAIN.subsplit(tfds.percent[-50:]), 
        batch_size=flags.batch_size,e = 0.1)
    
test_ds = _get_dataset('mnist', model, 
        split=tfds.Split.TEST, batch_size=flags.batch_size, 
        e = 0.9)
# Build environments
envs = [
    train_ds1,
    train_ds2,
    test_ds
  ]
# Define loss function helpers
# not possible to use tf.keras.losses.SparseCategoricalCrossentropy due to:
# https://github.com/tensorflow/tensorflow/issues/27875
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False) 
def mean_nll(logits, y):
    return loss_object(y, logits)

def mean_accuracy(logits, y):
    preds = tf.cast(logits > 0., dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.abs(preds - y) < 1e-2, dtype=tf.float32))
    return accuracy

def penalty(logits, y):
    with tf.GradientTape() as tape_src:
        scale = tf.ones(1,1)
        tape_src.watch(scale)
        loss = mean_nll(logits * scale, y)
        grad = tape_src.gradient(loss, scale)
    return tf.reduce_sum(grad**2)

# define optimizer
optimizer = tf.keras.optimizers.Adam(lr=flags.lr, epsilon=1e-08)

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
    for env0, env1, env2 in zip(envs[0], envs[1], envs[2]):
        with tf.GradientTape() as tape_src:
            env = [[], [], []]
            env[0].append(mean_nll(model(env0["image"]), env0["label"]))
            env[0].append(mean_accuracy(model(env0["image"]), env0["label"]))
            env[0].append(penalty(model(env0["image"]), env0["label"]))
            env[1].append(mean_nll(model(env1["image"]), env1["label"]))
            env[1].append(mean_accuracy(model(env1["image"]), env1["label"]))
            env[1].append(penalty(model(env1["image"]), env1["label"]))
            env[2].append(mean_nll(model(env2["image"]), env2["label"]))
            env[2].append(mean_accuracy(model(env2["image"]), env2["label"]))
            env[2].append(penalty(model(env2["image"]), env2["label"]))
            

            train_nll = tf.reduce_mean([env[0][0], env[1][0]])
            train_accuracy = tf.reduce_mean([env[0][1], env[1][1]])
            train_penalty = tf.reduce_mean([env[0][2], env[1][2]])
            test_accuracy = env[2][1]
            
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
                if step >= flags.penalty_anneal_iters else 100.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            # update weights of classifier
            grads = tape_src.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if step == 0:    
        pretty_print('epoch', 'train nll', 'train acc', 'test acc')
    # if step % 10 == 0:
    pretty_print(np.int32(step+1),
                train_loss.result().numpy(),
                (train_acc.result()*100).numpy(),
                (test_acc.result()*100).numpy())
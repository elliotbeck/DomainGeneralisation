import argparse
import copy
import datetime
import json
import pickle
import math
import os
import random
from random import shuffle
import matplotlib
import sklearn
import itertools

#matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from models import get_model, CrossGrad
from util import copy_source

plt.interactive(False)

from absl import flags, app, logging
import tensorflow as tf
#import tensorflow_transform as tft
import numpy as np
import time
#from datasets import pacs
import experiment_repo as repo

import util
import local_settings
from collections import defaultdict

DEBUG = False


parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--config', type=str, 
    default="configs/config_class_crossgrad.json",
    help='Path to config file.')
parser.add_argument('--all_checkpoints_folder', type=str, 
    default="checkpoints_pretr", help='Checkpoint folder name.')
parser.add_argument('--reload_ckpt', type=str, default="None", 
    help='Run ID from which to continue training.')
parser.add_argument('--local_json_dir_name', type=str,
    help='Folder name to save results jsons.')  
parser.add_argument('--dataset', type=str, help='Dataset.')
parser.add_argument('--name', type=str, help='Model name.')
parser.add_argument('--learning_rate', type=float, help='Learning rate.') 
parser.add_argument('--batch_size', type=int, help='Batch size.')
parser.add_argument('--num_epochs', type=int, help='Number of epochs.')
parser.add_argument('--decay_every', type=float, help='Decay steps.')
parser.add_argument('--img_size', type=int, help='Number of epochs.')
parser.add_argument('--l2_penalty_weight', type=float, help='L2 penalty weight.')
parser.add_argument('--validation_size', type=int, help='validation set size.')
parser.add_argument('--overwrite_configs', type=int, 
    help='Flag whether to overwrite configs.')
parser.add_argument('--dropout_rate', type=float, help='Dropout rate.')
parser.add_argument('--use_dropout', type=int, help='Flag whether to use dropout.')
parser.add_argument('--alpha', type=float, help='weighting factor of classification loss.')
parser.add_argument('--lambda', type=float, help='weighting factor of generator.')
parser.add_argument('--seed', type=int, help='Seed.')



def loss_fn_domain(features1, features2, model_domain, config, training):
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    domain1 = tf.squeeze(features1["domain"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    domain2 = tf.squeeze(features2["domain"])

    domain_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(domain1, axis=-1, 
                                depth=config.num_classes_domain), 
                                logits = model_domain(inputs1, training=training)), name='domain_loss1')
    domain_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(domain2, axis=-1, 
                                depth=config.num_classes_domain), 
                                logits = model_domain(inputs2, training=training)), name='domain_loss2')
    domain_loss = tf.reduce_mean([domain_loss1,domain_loss2])
    return domain_loss

def loss_fn_label(features1, features2, model_label, config, training):
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])

    # L2 regularizers
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in 
        model_label.trainable_variables if 'bias' not in v.name])

    model_label_output1 = model_label(inputs1, training=training)
    model_label_output2 = model_label(inputs2, training=training)


    label_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(label1, axis=-1, 
                                depth=config.num_classes_label), 
                                logits = model_label_output1), name='Label_loss1')
    label_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(label2, axis=-1, 
                                depth=config.num_classes_label), 
                                logits = model_label_output2), name='Label_loss2')                         
    label_loss = tf.reduce_mean([label_loss1,label_loss2])

    accuracy1 = tf.reduce_mean(
        tf.where(tf.equal(label1, tf.argmax(model_label_output1, axis=-1)),
                    tf.ones_like(label1, dtype=tf.float32),
                    tf.zeros_like(label1, dtype=tf.float32)))

    accuracy2 = tf.reduce_mean(
        tf.where(tf.equal(label2, tf.argmax(model_label_output2, axis=-1)),
                    tf.ones_like(label2, dtype=tf.float32),
                    tf.zeros_like(label2, dtype=tf.float32)))

    accuracy = tf.reduce_mean([accuracy1, accuracy2])
    return label_loss, accuracy, l2_regularizer

def _train_step(model_label, model_domain, features1, features2,
                optimizer, global_step, config):
    with tf.GradientTape(persistent=True) as tape_src:

        tape_src.watch(features1["image"])
        tape_src.watch(features2["image"])

        # get loss of labels
        mean_classification_loss, accuracy, l2_regularizer = loss_fn_label(
            features1, features2, model_label ,config=config, training=True)

        tf.summary.scalar("binary_crossentropy", mean_classification_loss, 
            step=global_step)
        tf.summary.scalar("accuracy", accuracy, step=global_step)

        total_loss = mean_classification_loss + \
            config.l2_penalty_weight*l2_regularizer
        # get loss of domains
        loss_domain = loss_fn_domain(features1, features2, model_domain, config, training=True)

        # get gradients wrt to inputs
        grads11 = tape_src.gradient(total_loss, features1["image"])
        grads12 = tape_src.gradient(total_loss, features2["image"])
        grads21 = tape_src.gradient(loss_domain, features1["image"])
        grads22 = tape_src.gradient(loss_domain, features2["image"])

        # create the new features as defined in the paper
        X_d1, X_d2 = {}, {}
        X_l1, X_l2 = {}, {}


        X_l1["image"] = features1["image"] + 0.5*grads11
        X_l1["label"] = features1["label"]
        X_l1["domain"] = features1["domain"]
        X_l2["image"] = features2["image"] + 0.5*grads12
        X_l2["label"] = features2["label"]
        X_l2["domain"] = features2["domain"]
        X_d1["image"] = features1["image"] + 0.5*grads21
        X_d1["label"] = features1["label"]
        X_d1["domain"] = features1["domain"]
        X_d2["image"] = features2["image"] + 0.5*grads22
        X_d2["label"] = features2["label"]
        X_d2["domain"] = features2["domain"]

    with tf.GradientTape(persistent=True) as tape_src:
        # calculate the losses with peturbated x
        loss_l, _, _ = loss_fn_label(X_d1, X_d2, model_label, config=config, training=True)
        loss_d = loss_fn_domain(X_l1, X_l2, model_domain, config=config, training=True)

        # calculate gradients for both neural nets
        grads3 = tape_src.gradient(0.9*loss_domain+0.1*loss_d, model_domain.trainable_variables)
        grads4 = tape_src.gradient(0.9*total_loss+0.1*loss_l, model_label.trainable_variables)

        # apply gradient to both neural nets
        optimizer.apply_gradients(zip(grads3, model_domain.trainable_variables))
        optimizer.apply_gradients(zip(grads4, model_label.trainable_variables))



def train_one_epoch(model_domain, model_label,  train_input1, train_input2,
                    optimizer,  global_step, config):
    train_input1.shuffle(buffer_size=10000)
    train_input2.shuffle(buffer_size=10000)
    for _input1, _input2 in zip(train_input1, train_input2):
        _train_step(model_label, model_domain, _input1, _input2, optimizer, global_step, config)


# compute the mean of all examples for a specific set (eval, validation, out-of-distribution, etc)
def eval_one_epoch(model_label, dataset, summary_directory, global_step, config, training):
    classification_loss = tf.metrics.Mean("binary_crossentropy")
    accuracy = tf.metrics.Mean("accuracy")

    dataset1, dataset2 = dataset.shard(2, 0), dataset.shard(2, 1)
    # losses = []
    # accuracies = []
    for _input1, _input2 in zip(dataset1,dataset2):
        _classification_loss, _accuracy, _ = loss_fn_label(
        features1=_input1, features2=_input2, model_label = model_label, 
        config=config, training=training)
        # losses.append(_classification_loss.numpy())
        # accuracies.append(_accuracy.numpy())

        # update mean-metric
        classification_loss(_classification_loss)
        accuracy(_accuracy)

    writer = tf.summary.create_file_writer(summary_directory)
    with writer.as_default(), tf.summary.record_if(True):
        tf.summary.scalar("classification_loss", classification_loss.result(), 
            step=global_step)
        tf.summary.scalar("accuracy", accuracy.result(), step=global_step)

    results_dict = {"accuracy": accuracy.result(), 
        "loss": classification_loss.result()}

    return results_dict



def _preprocess_exampe(model_label, example, dataset_name, domain,e):
    example["image"] = tf.cast(example["image"], dtype=tf.float64)/255.
    # 2x subsample for computational convenience
    example["image"] = example["image"][::2, ::2, :]
    example["image"] = tf.squeeze(example["image"], axis=-1)
    # Assign a binary label based on the digit; flip label with probability 0.25
    label = tf.cast([[example["label"] < 5]], dtype=tf.int64)
    label = util.tf_xor(label, tf.cast(util.tf_bernoulli(0.25, 1), dtype=tf.int64))
    # Assign a color based on the label; flip the color with probability e
    colors = util.tf_xor(label, tf.cast(util.tf_bernoulli(e, 1), dtype=tf.int64))
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
    example["label"] = label
    example["domain"] = domain

    return example   

def _get_dataset(dataset_name, model_label, split, batch_size, e, domain,
    num_batches=None):

    dataset, info = tfds.load(dataset_name, data_dir=local_settings.TF_DATASET_PATH, 
        split=split, with_info=True)
    dataset = dataset.map(lambda x: _preprocess_exampe(model_label, x, dataset_name, domain,e))
    dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    if num_batches is not None:
        dataset = dataset.take(num_batches)

    # dataset = dataset.prefetch(2)

    return dataset


def main():
    # parse args and get configs
    args = parser.parse_args()
    logging.set_verbosity(logging.INFO)
    random.seed(args.seed)
    # reload model from checkpoint or train from scratch
    if args.reload_ckpt != "None":
        checkpoint_path = os.path.join(local_settings.MODEL_PATH, 
            args.all_checkpoints_folder)
        checkpoint_folders = os.listdir(checkpoint_path)
        checkpoint_folder = [f for f in checkpoint_folders if args.reload_ckpt in f]
        if len(checkpoint_folder) == 0:
            raise Exception("No matching folder found.")
        elif len(checkpoint_folder) > 1:
            logging.info(checkpoint_folder)
            raise Exception("More than one matching folder found.")
        else:
            checkpoint_folder = checkpoint_folder[0]
            logging.info("Restoring from {}".format(checkpoint_folder))
        checkpoint_dir = os.path.join(checkpoint_path, checkpoint_folder)
        
        if not args.overwrite_configs:
            # reload configs from file
            with open(os.path.join(checkpoint_dir, "hparams.pkl"), 'rb') as f:
                config_dict = pickle.load(f)
        else:
            # get configs
            config_dict = util.get_config(args.config)
            config_dict = util.update_config(config_dict, args)
    else:
        # get configs
        config_dict = util.get_config(args.config)
        config_dict = util.update_config(config_dict, args)

    config_dict_copy = copy.deepcopy(config_dict)
    config = util.config_to_namedtuple(config_dict)

    # Initialize the repo
    logging.info("==> Creating repo..")
    exp_repo = repo.ExperimentRepo(local_dir_name=config.local_json_dir_name,
        root_dir=local_settings.MODEL_PATH)

    if args.reload_ckpt != "None":
        exp_id = config_dict["id"]
    else:
        exp_id = None
    
    # Create new experiment
    exp_id = exp_repo.create_new_experiment(config.dataset, 
        config_dict_copy, exp_id)
    config_dict_copy["id"] = exp_id

    # Set up model directory
    current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
    ckpt_dir_name = args.all_checkpoints_folder if not DEBUG else 'checkpoints_tmp'
    ckpt_dir = os.path.join(local_settings.MODEL_PATH, ckpt_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.reload_ckpt != "None":
        model_dir = checkpoint_dir
    else:
        model_dir = os.path.join(
            ckpt_dir, "ckpt_{}_{}".format(current_time, exp_id))
    
    # Save hyperparameter settings
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "hparams.json")):
        with open(os.path.join(model_dir, "hparams.json"), 'w') as f:
            json.dump(config_dict_copy, f, indent=2, sort_keys=True)
        with open(os.path.join(model_dir, "hparams.pkl"), 'wb') as f:
            pickle.dump(config_dict_copy, f)

    # Set optimizers
    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    #     config.learning_rate, config.decay_every, 
    #     config.decay_base, staircase=True)

    
    # learning rate = 0.02 in paper
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.02)



    if args.reload_ckpt != "None":
        # TODO: fix this hack
        epoch_start = int(sorted([f for f in os.listdir(checkpoint_dir) 
            if 'ckpt-' in f])[-1].split('ckpt-')[1].split('.')[0])
        init_gs = 0
    else:
        epoch_start = 0
        init_gs = 0

    global_step = tf.Variable(initial_value=init_gs, name="global_step", 
        trainable=False, dtype=tf.int64)

    # Get model
    model_domain = get_model(config.name_classifier_domain, config)
    model_label = get_model(config.name_classifier_label, config)

    # Get datasets
    if DEBUG:
        num_batches = 5
    else:
        num_batches = None

    ds_train1 = _get_dataset(config.dataset, model_label,
        split=tfds.Split.TRAIN.subsplit(tfds.percent[:50]), 
        batch_size=tf.cast(config.batch_size/2, tf.int64), 
        num_batches=num_batches, domain = tf.constant(0), e = 0.2)

    ds_train2 = _get_dataset(config.dataset, model_label, 
        split=tfds.Split.TRAIN.subsplit(tfds.percent[-50:]),
        batch_size=tf.cast(config.batch_size/2, tf.int64),
        num_batches=num_batches, domain = tf.constant(1), e = 0.1)
    
    ds_val = _get_dataset(config.dataset, model_label, 
        split=tfds.Split.TEST, batch_size=config.batch_size, 
        num_batches=num_batches, domain = tf.constant(2), e = 0.9)

    # TODO: add test set - done
    
    show_inputs = iter(ds_train1)
    _ = model_label(next(show_inputs)["image"])

    # Set up checkpointing
    if args.reload_ckpt != "None":
        ckpt = tf.train.Checkpoint(model=model_label, global_step=global_step)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_consumed()
    else:
        ckpt = tf.train.Checkpoint(model=model_label, global_step=global_step)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3) 

    writer = tf.summary.create_file_writer(manager._directory)
    with writer.as_default(), tf.summary.record_if(lambda: int(global_step.numpy()) % 100 == 0):
        for epoch in range(epoch_start, config.num_epochs):
            
            start_time = time.time()

            # random = np.array([0, 1, 2])
            # np.random.shuffle(random)
            # rand_inputs = [ds_train1, ds_train2, ds_train3]


            train_one_epoch(model_domain=model_domain, model_label=model_label,
                train_input1=ds_train1, train_input2=ds_train2, 
                optimizer=optimizer, global_step=global_step, config=config)

            train1_metr = eval_one_epoch(model_label=model_label, dataset=ds_train1,
                summary_directory=os.path.join(manager._directory, "train1"), 
                global_step=global_step, config=config, training=False)

            train2_metr = eval_one_epoch(model_label=model_label, dataset=ds_train2,
                summary_directory=os.path.join(manager._directory, "train2"), 
                global_step=global_step, config=config, training=False)
            
            val_metr = eval_one_epoch(model_label=model_label, dataset=ds_val,
                summary_directory=os.path.join(manager._directory, "val"), 
                global_step=global_step, config=config, training=False)
           
            # if epoch == (config.num_epochs - 1):
            #     # full training set
            #     train_metr = eval_one_epoch(model_classifier=model_classifier, dataset=ds_train_complete,
            #         summary_directory=os.path.join(manager._directory, "train"), 
            #         global_step=global_step, config=config, training=False)
            #     # full test_out set
            #     test_out_metr = eval_one_epoch(model_classifier=model_classifier, dataset=ds_val_out,
            #         summary_directory=os.path.join(manager._directory, "val_out"),
            #         global_step=global_step, config=config, training=False)
            #     # full test_in set
            #     test_in_metr = eval_one_epoch(model_classifier=model_classifier, dataset=ds_val_in,
            #         summary_directory=os.path.join(manager._directory, "val_in"),
            #         global_step=global_step, config=config, training=False)


            manager.save()

            logging.info("\n #### \n epoch: %d, time: %0.2f" % 
                (epoch, time.time() - start_time))
            logging.info("Global step: {}".format(global_step.numpy()))
            logging.info("train1_accuracy: {:2f}, train1_loss: {:4f}".format(
                train1_metr['accuracy'], train1_metr['loss']))
            logging.info("train2_accuracy: {:2f}, train2_loss: {:4f}".format(
                train2_metr['accuracy'], train2_metr['loss']))                
            logging.info("val_accuracy: {:2f}, val_loss: {:4f}".format(
                val_metr['accuracy'], val_metr['loss']))

            if epoch == epoch_start:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                copy_source(dir_path, manager._directory)

    
    # Mark experiment as completed
    # TODO: add other metrics - done
    exp_repo.mark_experiment_as_completed(exp_id, 
        train1_accuracy=train1_metr['accuracy'],
        train2_accuracy=train2_metr['accuracy'],
        val_accuracy=val_metr['accuracy'])

if __name__ == "__main__":
    main()
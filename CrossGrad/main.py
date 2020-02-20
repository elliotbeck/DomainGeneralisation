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
from datasets import pacs
import experiment_repo as repo

import util
import local_settings
from collections import defaultdict

DEBUG = True


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
parser.add_argument('--epsL', type=float, help='Multiple for labels.')
parser.add_argument('--epsD', type=float, help='Multiple for domains.')
parser.add_argument('--seed', type=int, help='Seed.')

# loss function for the domain network
def loss_fn_domain(features1, features2, features3, model_domain, config, training):
    # get data
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    domain1 = tf.squeeze(features1["domain"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    domain2 = tf.squeeze(features2["domain"])
    inputs3 = features3["image"]
    label3 = tf.squeeze(features3["label"])
    domain3 = tf.squeeze(features3["domain"])

    # calculate loss per source domain
    domain_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(domain1, axis=-1, 
                                depth=config.num_classes_domain), 
                                logits = model_domain(inputs1, training=training)), name='domain_loss1')
    domain_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(domain2, axis=-1, 
                                depth=config.num_classes_domain), 
                                logits = model_domain(inputs2, training=training)), name='domain_loss2')
    domain_loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(domain3, axis=-1, 
                                depth=config.num_classes_domain), 
                                logits = model_domain(inputs3, training=training)), name='domain_loss3')
    # get mean of all source domain losses                             
    domain_loss = tf.reduce_mean([domain_loss1, domain_loss2, domain_loss3])
    return domain_loss

# loss function for the class network
def loss_fn_label(features1, features2, features3, model_label, config, training):
    # get data
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    inputs3 = features3["image"]
    label3 = tf.squeeze(features3["label"])

    # L2 regularizers
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in 
        model_label.trainable_variables if 'bias' not in v.name])

    # get predictions on all source domains
    model_label_output1 = model_label(inputs1, training=training)
    model_label_output2 = model_label(inputs2, training=training)
    model_label_output3 = model_label(inputs3, training=training)

    # get label loss per domain
    label_loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label1,
                                logits = model_label_output1))

    label_loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label2, 
                                logits = model_label_output2))
    label_loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label3, 
                                logits = model_label_output3))  

    # get mean loss over all domains                        
    label_loss = tf.reduce_mean([label_loss1,label_loss2,label_loss3])

    # calculate accuracy per source domain
    accuracy1 = tf.reduce_mean(
        tf.where(tf.equal(label1, tf.argmax(model_label_output1, axis=-1)),
                    tf.ones_like(label1, dtype=tf.float32),
                    tf.zeros_like(label1, dtype=tf.float32)))

    accuracy2 = tf.reduce_mean(
        tf.where(tf.equal(label2, tf.argmax(model_label_output2, axis=-1)),
                    tf.ones_like(label2, dtype=tf.float32),
                    tf.zeros_like(label2, dtype=tf.float32)))
    
    accuracy3 = tf.reduce_mean(
        tf.where(tf.equal(label3, tf.argmax(model_label_output3, axis=-1)),
                    tf.ones_like(label3, dtype=tf.float32),
                    tf.zeros_like(label3, dtype=tf.float32)))
    # get mean accuracy over all source domains
    accuracy = tf.reduce_mean([accuracy1, accuracy2, accuracy3])

    return label_loss, accuracy, l2_regularizer

def _train_step(model_label, model_domain, features1, features2, features3,
                optimizer1, optimizer2, global_step, config):
    with tf.GradientTape(persistent=True) as tape_src:

        tape_src.watch(features1["image"])
        tape_src.watch(features2["image"])
        tape_src.watch(features3["image"])

        # get loss of labels
        mean_classification_loss, accuracy, l2_regularizer = loss_fn_label(
            features1, features2, features3, model_label ,config=config, training=True)

        tf.summary.scalar("binary_crossentropy", mean_classification_loss, 
            step=global_step)
        tf.summary.scalar("accuracy", accuracy, step=global_step)

        total_loss = mean_classification_loss + \
            config.l2_penalty_weight*l2_regularizer
        # get loss of domains
        loss_domain = loss_fn_domain(features1, features2, features3, model_domain, config, training=True)

        # get gradients wrt to inputs
        grads11 = tape_src.gradient(total_loss, features1["image"])
        grads12 = tape_src.gradient(total_loss, features2["image"])
        grads13 = tape_src.gradient(total_loss, features3["image"])
        grads21 = tape_src.gradient(loss_domain, features1["image"])
        grads22 = tape_src.gradient(loss_domain, features2["image"])
        grads23 = tape_src.gradient(loss_domain, features3["image"])

        # create the new features as defined in the paper
        X_d1, X_d2, X_d3 = {}, {}, {}
        X_l1, X_l2, X_l3 = {}, {}, {}


        X_l1["image"] = features1["image"] + config.epsL*grads11
        X_l1["label"] = features1["label"]
        X_l1["domain"] = features1["domain"]
        X_l2["image"] = features2["image"] + config.epsL*grads12
        X_l2["label"] = features2["label"]
        X_l2["domain"] = features2["domain"]
        X_l3["image"] = features3["image"] + config.epsL*grads13
        X_l3["label"] = features3["label"]
        X_l3["domain"] = features3["domain"]
        X_d1["image"] = features1["image"] + config.epsD*grads21
        X_d1["label"] = features1["label"]
        X_d1["domain"] = features1["domain"]
        X_d2["image"] = features2["image"] + config.epsD*grads22
        X_d2["label"] = features2["label"]
        X_d2["domain"] = features2["domain"]
        X_d3["image"] = features3["image"] + config.epsD*grads23
        X_d3["label"] = features3["label"]
        X_d3["domain"] = features3["domain"]

    
    with tf.GradientTape(persistent=True) as tape_src:
        
        # get loss of domains
        loss_domain = loss_fn_domain(features1, features2, features3, 
                                        model_domain, config, training=True)

        # get loss of labels
        mean_classification_loss, accuracy, l2_regularizer = loss_fn_label(
            features1, features2, features3, model_label, config=config, training=True)
        total_loss = mean_classification_loss + \
            config.l2_penalty_weight*l2_regularizer
        # calculate the losses with peturbated x
        loss_l, _, _ = loss_fn_label(X_d1, X_d2, X_d3, model_label ,config=config, training=True)
        loss_d = loss_fn_domain(X_l1, X_l2, X_l3, model_domain ,config=config, training=True)

        # calculate the losses for the gradients
        loss3 = 0.9*loss_domain+0.1*loss_d
        loss4 = 0.9*total_loss+0.1*loss_l

        # calculate gradients for both neural nets
        grads3 = tape_src.gradient(loss3, model_domain.trainable_variables)
        grads4 = tape_src.gradient(loss4, model_label.trainable_variables)

        # apply gradient to both neural nets
        optimizer1.apply_gradients(zip(grads3, model_domain.trainable_variables))
        optimizer2.apply_gradients(zip(grads4, model_label.trainable_variables))



def train_one_epoch(model_domain, model_label, train_input1, train_input2, train_input3,
                    optimizer1, optimizer2, global_step, config):
    train_input1.shuffle(buffer_size=10000)
    train_input2.shuffle(buffer_size=10000)
    train_input3.shuffle(buffer_size=10000)
    for _input1, _input2, _input3 in zip(train_input1, train_input2, train_input3):
        _train_step(model_label, model_domain, _input1, _input2, _input3, optimizer1, 
        optimizer2, global_step, config)


# compute the mean of all examples for a specific set (eval, validation, out-of-distribution, etc)
def eval_one_epoch(model_label, dataset, summary_directory, global_step, config, training):
    classification_loss = tf.metrics.Mean("binary_crossentropy")
    accuracy = tf.metrics.Mean("accuracy")

    dataset1, dataset2, dataset3 = dataset.shard(3, 0), dataset.shard(3, 1), dataset.shard(3, 2)
    # losses = []
    # accuracies = []
    for _input1, _input2, _input3 in zip(dataset1,dataset2,dataset3):
        _classification_loss, _accuracy, _ = loss_fn_label(
        features1=_input1, features2=_input2, features3=_input3, model_label = model_label, 
        config=config, training=training)
        # losses.append(_classification_loss.numpy())
        # accuracies.append(_accuracy.numpy())

        # print pictures
        with tf.GradientTape(persistent=True) as tape_src:

            tape_src.watch(_input1["image"])
            tape_src.watch(_input2["image"])
            tape_src.watch(_input3["image"])

            # get loss of labels
            mean_classification_loss, accuracy, l2_regularizer = loss_fn_label(
                _input1, _input2, _input3, model_label ,config=config, training=True)


            total_loss = mean_classification_loss + \
                config.l2_penalty_weight*l2_regularizer


            # get gradients wrt to inputs
            grads11 = tape_src.gradient(total_loss, _input1["image"])
            grads12 = tape_src.gradient(total_loss, _input2["image"])
            grads13 = tape_src.gradient(total_loss, _input3["image"])


            # create the new features as defined in the paper
            X_l1, X_l2, X_l3 = {}, {}, {}


            X_l1["image"] = _input1["image"] + config.epsL*grads11
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/fake1.png', X_l1["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/original1.png', _input1["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/peturbation1.png', X_l1["image"][0]-_input1["image"][0])

            X_l2["image"] = _input2["image"] + config.epsL*grads12
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/fake2.png', X_l2["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/original2.png', _input2["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/peturbation2.png', X_l2["image"][0]-_input2["image"][0])

            X_l3["image"] = _input3["image"] + config.epsL*grads13
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/fake3.png', X_l3["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/original3.png', _input3["image"][0])
            plt.imsave('/cluster/home/ebeck/DomainGeneralisation/CrossGrad/images/peturbation3.png', X_l3["image"][0]-_input3["image"][0])

        _classification_loss, _accuracy, _ = loss_fn_label(
            features1=_input1, features2=_input2, features3=_input3, model_label = model_label, 
            config=config, training=training)

        # update mean-metric
        classification_loss(_classification_loss)
        print(_accuracy)
        accuracy(_accuracy)

    writer = tf.summary.create_file_writer(summary_directory)
    with writer.as_default(), tf.summary.record_if(True):
        tf.summary.scalar("classification_loss", classification_loss.result(), 
            step=global_step)
        tf.summary.scalar("accuracy", accuracy.result(), step=global_step)

    results_dict = {"accuracy": accuracy.result(), 
        "loss": classification_loss.result()}

    return results_dict


def _preprocess_exampe(model_label, example, dataset_name, config):
    example["image"] = tf.cast(example["image"], dtype=tf.float64)/255.
    example["image"] = tf.image.resize(example["image"], size=(model_label.input_shape[0], model_label.input_shape[1]))
    example["label"] = example["attributes"]["label"]
    example["domain"] = example["attributes"]["domain"]
    # encode source domains as 0,1,2
    domain = example["domain"]
    if domain == config.training_domains[0]:
        example["domain"] = tf.constant(0)
    elif domain == config.training_domains[1]:
        example["domain"] = tf.constant(1)
    else:
        example["domain"] = tf.constant(2)
    example["label"] = tf.subtract(example["label"],1)
    return example


def _get_dataset(dataset_name, model_label, validation_split, split, batch_size, config, 
    num_batches=None):

    builder_kwargs = {
        "validation_split": validation_split,
    }

    dataset, _ = tfds.load(dataset_name, data_dir=local_settings.TF_DATASET_PATH, 
        split=split, builder_kwargs=builder_kwargs, with_info=True)
    dataset = dataset.map(lambda x: _preprocess_exampe(model_label, x, dataset_name, config))
    #dataset = dataset.shuffle(512)
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
    optimizer1 = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.0)
    optimizer2 = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.0)



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

    ds_train_complete = _get_dataset(config.dataset, model_label, config.test_domain,
        split=tfds.Split.TRAIN, batch_size=tf.cast(config.batch_size/3, tf.int64), config = config,
        num_batches=num_batches)

    ds_train1 = _get_dataset(config.dataset, model_label, config.test_domain,
        split="train1", batch_size=tf.cast(config.batch_size/3, tf.int64), config = config,
        num_batches=num_batches)
    
    ds_train2 = _get_dataset(config.dataset, model_label, config.test_domain, config = config,
        split="train2", batch_size=tf.cast(config.batch_size/3, tf.int64), 
        num_batches=num_batches)

    ds_train3 = _get_dataset(config.dataset, model_label, config.test_domain, config = config,
        split="train3", batch_size=tf.cast(config.batch_size/3, tf.int64),
        num_batches=num_batches)

    ds_val_in = _get_dataset(config.dataset, model_label, config.test_domain, config = config,
        split="val_in", batch_size=tf.cast(config.batch_size/3, tf.int64),
        num_batches=num_batches)

    ds_val_out = _get_dataset(config.dataset, model_label, config.test_domain, config = config,
        split="val_out", batch_size=tf.cast(config.batch_size/3, tf.int64),
        num_batches=num_batches)

    ds_test_in = _get_dataset(config.dataset, model_label, config.test_domain, config = config,
        split="test_in", batch_size=tf.cast(config.batch_size/3, tf.int64), 
        num_batches=num_batches)

    ds_test_out = _get_dataset(config.dataset, model_label, config.test_domain, config = config, 
        split="test_out", batch_size=tf.cast(config.batch_size/3, tf.int64),
        num_batches=num_batches)

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

            train_one_epoch(model_domain = model_domain, model_label=model_label, train_input1=ds_train1, 
                train_input2=ds_train2, train_input3=ds_train3, optimizer1=optimizer1, optimizer2=optimizer2, 
                global_step=global_step, config=config)

            train_metr = eval_one_epoch(model_label=model_label, dataset=ds_train_complete,
                summary_directory=os.path.join(manager._directory, "train"), 
                global_step=global_step, config=config, training=False)
            
            val_out_metr = eval_one_epoch(model_label=model_label, dataset=ds_val_out,
                summary_directory=os.path.join(manager._directory, "val_out"), 
                global_step=global_step, config=config, training=False)

            val_in_metr = eval_one_epoch(model_label=model_label, dataset=ds_val_in,
                summary_directory=os.path.join(manager._directory, "val_in"),
                global_step=global_step, config=config, training=False)

            test_in_metr = eval_one_epoch(model_label=model_label, dataset=ds_test_in,
                summary_directory=os.path.join(manager._directory, "test_in"), 
                global_step=global_step, config=config, training=False)

            test_out_metr = eval_one_epoch(model_label=model_label, dataset=ds_test_out,
                summary_directory=os.path.join(manager._directory, "test_out"),
                global_step=global_step, config=config, training=False)


            manager.save()

            logging.info("\n #### \n epoch: %d, time: %0.2f" % 
                (epoch, time.time() - start_time))
            logging.info("Global step: {}".format(global_step.numpy()))
            logging.info("train_accuracy: {:2f}, train_loss: {:4f}".format(
                train_metr['accuracy'], train_metr['loss']))
            logging.info("val_out_accuracy: {:2f}, val_out_loss: {:4f}".format(
                val_out_metr['accuracy'], val_out_metr['loss']))
            logging.info("val_in_accuracy: {:2f}, val_in_loss: {:4f}".format(
                val_in_metr['accuracy'], val_in_metr['loss']))
            logging.info("test_in_accuracy: {:2f}, test_in_loss: {:4f}".format(
                test_in_metr['accuracy'], test_in_metr['loss']))
            logging.info("test_out_accuracy: {:2f}, test_out_loss: {:4f}".format(
                test_out_metr['accuracy'], test_out_metr['loss']))

            if epoch == epoch_start:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                copy_source(dir_path, manager._directory)

    
    # Mark experiment as completed
    # TODO: add other metrics - done
    exp_repo.mark_experiment_as_completed(exp_id, 
        train_accuracy=train_metr['accuracy'],
        val_out_accuracy=val_out_metr['accuracy'],
        val_in_accuracy=val_in_metr['accuracy'],
        test_in_accuracy=test_in_metr['accuracy'],
        test_out_accuracy=test_out_metr['accuracy'])

if __name__ == "__main__":
    main()
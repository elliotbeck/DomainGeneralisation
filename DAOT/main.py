import argparse
import copy
import datetime
import json
import pickle
import math
import os
from random import shuffle
import matplotlib
import sklearn
import itertools

#matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from models import get_model
from util import copy_source

plt.interactive(False)

from absl import flags, app, logging
import tensorflow as tf
import numpy as np
import time
from datasets import pacs
import experiment_repo as repo

import util
import local_settings

DEBUG = False

parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--config', type=str, 
    default="configs/config_class.json",
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

# loss funtion for classifier
def loss_fn_classifier(model_classifier, model_generator, features1, features2, config, training):
    # save features and labels from the two random training domains and concat them
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    inputs = tf.concat([inputs1, inputs2], 0)
    label = tf.concat([label1, label2], 0)
    # get generated inputs, labels stay the same
    inputs_generated = model_generator(inputs, training=False)
    label_generated = label
    #inputs_all = tf.concat([inputs, inputs_generated], 0)
    label_all = tf.concat([label, label_generated], 0)

    # L2 regularizers
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in 
        model_classifier.trainable_variables if 'bias' not in v.name])
    # get label predictions
    model_classifier_output_original = model_classifier(inputs, training=training)
    model_classifier_output_generated = model_classifier(inputs_generated, 
                                            training=training)
    # get mean classification loss on original data                                        
    classification_loss_original = tf.losses.binary_crossentropy(
        tf.one_hot(label, axis=-1, depth=config.num_classes),
        model_classifier_output_original, from_logits=False)
    mean_classification_loss_original = tf.math.reduce_mean(classification_loss_original)
    # get mean classification loss on generated data
    classification_loss_generated = tf.losses.binary_crossentropy(
        tf.one_hot(label_generated, axis=-1, depth=config.num_classes),
        model_classifier_output_generated, from_logits=False)
    mean_classification_loss_generated = tf.math.reduce_mean(classification_loss_generated)
    # get weighted total loss
    classification_loss = classification_loss_original + classification_loss_generated
    mean_classification_loss_weighted = (1-config.alpha) * mean_classification_loss_original + \
        config.alpha * mean_classification_loss_generated
    # calculate accuracy 
    accuracy = tf.math.reduce_mean(
        tf.where(tf.equal(label, tf.argmax(model_classifier_output_original, axis=-1)),
                    tf.ones_like(label, dtype=tf.float32),
                    tf.zeros_like(label, dtype=tf.float32)))
    return mean_classification_loss_weighted, l2_regularizer, accuracy, classification_loss

# loss function for generator
def loss_fn_generator(model_classifier, model_critic, model_generator, features1, 
        features2, config, training):
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    label_generated1 = label1
    label_generated2 = label2

    X_generated1 = model_generator(inputs1, training=training)
    X_generated2 = model_generator(inputs2, training=training)
    X_critic_true1 = model_critic(inputs1, training=False)
    X_critic_true2 = model_critic(inputs2, training=False)
    X_critic_generated1 = model_critic(X_generated1, training=False)
    X_critic_generated2 = model_critic(X_generated2, training=False)

    # get label predictions
    model_classifier_output_generated1 = model_classifier(X_generated1, 
                                            training=False)
    model_classifier_output_generated2 = model_classifier(X_generated2, 
                                            training=False)

    # get mean classification loss on generated data
    classification_loss_generated = tf.losses.binary_crossentropy(
        tf.one_hot(tf.concat([label_generated1, label_generated2], 0), axis=-1, depth=config.num_classes),
        tf.concat([model_classifier_output_generated1, model_classifier_output_generated2], 0), from_logits=False)
    mean_classification_loss_generated = tf.math.reduce_mean(classification_loss_generated)
    
    # compute sinkhorn distances for M1
    # in case computational complexity doesn't matter replace zip with itertools.product
    sinkhorn_dist_intra1 = []
    for _input1, _input2 in itertools.product(X_critic_true1, X_critic_generated1):
        # compute M1 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_intra1.append(sinkhorn_dist)
    # compute sinkhorn distances for M2
    sinkhorn_dist_intra2 = []
    for _input1, _input2 in itertools.product(X_critic_true2, X_critic_generated2):
        # compute M2 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_intra2.append(sinkhorn_dist)
    
    sinkhorn_dist_intra = tf.math.reduce_sum(sinkhorn_dist_intra1)+tf.math.reduce_sum(sinkhorn_dist_intra2)
    # compute sinkhorn distances for M3
    sinkhorn_dist_inter1 = []
    for _input1, _input2 in itertools.product(X_critic_true2, X_critic_generated1):
        # compute M3 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_inter1.append(sinkhorn_dist)
        
    # compute sinkhorn distances for M4
    sinkhorn_dist_inter2 = []
    for _input1, _input2 in itertools.product(X_critic_true1, X_critic_generated2):
        # compute M4 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_inter2.append(sinkhorn_dist)

    sinkhorn_dist_inter = tf.math.reduce_sum(sinkhorn_dist_inter1)+tf.math.reduce_sum(sinkhorn_dist_inter2)
    loss_generator = mean_classification_loss_generated - sinkhorn_dist_intra - sinkhorn_dist_inter
    return -loss_generator


# loss function for critic
def loss_fn_critic(model_critic, model_generator, features1, features2, config, training):
    inputs1 = features1["image"]
    label1 = tf.squeeze(features1["label"])
    inputs2 = features2["image"]
    label2 = tf.squeeze(features2["label"])
    label_generated1 = label1
    label_generated2 = label2

    X_generated1 = model_generator(inputs1, training=False)
    #plt.imsave('/cluster/home/ebeck/DomainGeneralisation/DAOT/images/fake1.png', X_generated1[0])
    #plt.imsave('/cluster/home/ebeck/DomainGeneralisation/DAOT/images/original1.png', inputs1[0])
    X_generated2 = model_generator(inputs2, training=False)
    X_critic_true1 = model_critic(inputs1, training=training)
    X_critic_true2 = model_critic(inputs2, training=training)
    X_critic_generated1 = model_critic(X_generated1, training=training)
    X_critic_generated2 = model_critic(X_generated2, training=training)

    # # compute M1 (cost_matrix)
    # norms_true = tf.norm(X_critic_true1,2, axis=1)
    # norms_generated = tf.norm(X_critic_generated1,2, axis=1)
    # matrix_norms = tf.tensordot(norms_true,norms_generated, axes=0)
    # matrix_critic = tf.tensordot(X_critic_true1,tf.transpose(X_critic_generated1), axes=1)
    # cost_matrix = 1 - matrix_critic/matrix_norms
    
    # compute sinkhorn distances for M1
    sinkhorn_dist_intra1 = []
    for _input1, _input2 in itertools.product(X_critic_true1, X_critic_generated1):
        # compute M1 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_intra1.append(sinkhorn_dist)
    # compute sinkhorn distances for M2
    sinkhorn_dist_intra2 = []
    for _input1, _input2 in itertools.product(X_critic_true2, X_critic_generated2):
        # compute M2 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_intra2.append(sinkhorn_dist)
    
    sinkhorn_dist_intra = tf.math.reduce_sum(sinkhorn_dist_intra1)+tf.math.reduce_sum(sinkhorn_dist_intra2)
    # compute sinkhorn distances for M3
    sinkhorn_dist_inter = []
    for _input1, _input2 in itertools.product(X_critic_true1, X_critic_true2):
        # compute M3 (cost_matrix)
        cost_matrix = util.compute_cost_matrix(_input1, _input2)
        # calulate sinkhorn distance
        _, sinkhorn_dist = util.compute_optimal_transport(cost_matrix, _input1 ,_input2)
        sinkhorn_dist_inter.append(sinkhorn_dist)
    loss_critic = sinkhorn_dist_intra - tf.math.reduce_sum(sinkhorn_dist_inter)
    return loss_critic


def _train_step(model_classifier, model_generator, model_critic, features1, features2, 
                optimizer1, optimizer2, optimizer3, global_step, config):
    with tf.GradientTape() as tape_src:
        # get loss of classifier
        mean_classification_loss_weighted, l2_regularizer, accuracy, _ = loss_fn_classifier(
            model_classifier, model_generator, features1, features2, config=config, training=True)

        tf.summary.scalar("binary_crossentropy", mean_classification_loss_weighted, 
            step=global_step)
        tf.summary.scalar("accuracy", accuracy, step=global_step)

        total_loss = mean_classification_loss_weighted + \
            config.l2_penalty_weight*l2_regularizer
        
        # update weights of classifier
        grads = tape_src.gradient(total_loss, model_classifier.trainable_variables)
        optimizer1.apply_gradients(zip(grads, model_classifier.trainable_variables))

    with tf.GradientTape() as tape_src:
        # get loss of critic
        loss_critic = loss_fn_critic(model_critic, model_generator, features1, features2, config, training=True)

        # update weights of critic
        grads = tape_src.gradient(loss_critic, model_critic.trainable_variables)
        optimizer2.apply_gradients(zip(grads, model_critic.trainable_variables))

    with tf.GradientTape() as tape_src:
        # get loss of generator 
        loss_generator = loss_fn_generator(model_classifier, model_critic, model_generator, features1, 
            features2, config, training=True)

        # update weights of generator
        grads = tape_src.gradient(loss_generator, model_generator.trainable_variables)
        optimizer3.apply_gradients(zip(grads, model_generator.trainable_variables))

        global_step.assign_add(1)


def train_one_epoch(model_classifier, model_generator, model_critic, train_input1, train_input2,
                    optimizer1, optimizer2, optimizer3, global_step, config):
    
    for _input1, _input2 in zip(train_input1, train_input2):
        _train_step(model_classifier, model_generator, model_critic, _input1, _input2, optimizer1,
        optimizer2, optimizer3, global_step, config)


# compute the mean of all examples for a specific set (eval, validation, out-of-distribution, etc)
def eval_one_epoch(model_classifier, model_generator, dataset, summary_directory, global_step, config, training):
    classification_loss = tf.metrics.Mean("binary_crossentropy")
    accuracy = tf.metrics.Mean("accuracy")

    dataset1, dataset2 = dataset.shard(2, 0), dataset.shard(2, 1)
    # losses = []
    # accuracies = []
    for _input1, _input2 in zip(dataset1,dataset2):
        _, _, _accuracy, _classification_loss = loss_fn_classifier(model_classifier, model_generator,
        features1=_input1, features2=_input2, config=config, training=training)
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


def _preprocess_exampe(model_classifier, example, dataset_name):
    example["image"] = tf.cast(example["image"], dtype=tf.float32)/255.
    example["image"] = tf.image.resize(example["image"], 
        size=(model_classifier.input_shape[0], model_classifier.input_shape[1]))
    example["label"] = example["attributes"]["label"]
    example["domain"] = example["attributes"]["domain"]
    example["label"] = tf.subtract(example["label"],1)
    return example


def _get_dataset(dataset_name, model_classifier, validation_split, split, batch_size, 
    num_batches=None):

    builder_kwargs = {
        "validation_split": validation_split,
    }

    dataset, info = tfds.load(dataset_name, data_dir=local_settings.TF_DATASET_PATH, 
        split=split, builder_kwargs=builder_kwargs, with_info=True)
    dataset = dataset.map(lambda x: _preprocess_exampe(model_classifier, x, dataset_name))
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
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate, config.decay_every, 
        config.decay_base, staircase=True)
    optimizer1 = tf.keras.optimizers.Adam(learning_rate)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate)
    optimizer3 = tf.keras.optimizers.Adam(learning_rate)

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
    model_classifier = get_model(config.name_classifier, config)
    model_generator = get_model(config.name_generator, config)
    model_critic = get_model(config.name_critic, config)

    # Get datasets
    if DEBUG:
        num_batches = 5
    else:
        num_batches = None

    ds_train_complete = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split=tfds.Split.TRAIN, batch_size=tf.cast(config.batch_size/2, tf.int64), 
        num_batches=num_batches)

    ds_train1 = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split="train1", batch_size=tf.cast(config.batch_size/2, tf.int64),
        num_batches=num_batches)
    
    ds_train2 = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split="train2", batch_size=tf.cast(config.batch_size/2, tf.int64), 
        num_batches=num_batches)

    ds_train3 = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split="train3", batch_size=tf.cast(config.batch_size/2, tf.int64),
        num_batches=num_batches)

    # ds_val = _get_dataset(config.dataset, model_classifier, config.test_domain,
    #     split=tfds.Split.VALIDATION, batch_size=config.batch_size, 
    #     num_batches=num_batches)

    # ds_test = _get_dataset(config.dataset, model_classifier, config.test_domain,
    #     split=tfds.Split.TEST, batch_size=config.batch_size,
    #     num_batches=num_batches)

    ds_val_in = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split="val_in", batch_size=tf.cast(config.batch_size/2, tf.int64),
        num_batches=num_batches)

    ds_val_out = _get_dataset(config.dataset, model_classifier, config.test_domain,
        split="val_out", batch_size=tf.cast(config.batch_size/2, tf.int64),
        num_batches=num_batches)

    # TODO: add test set - done
    
    show_inputs = iter(ds_train1)
    _ = model_classifier(next(show_inputs)["image"])

    # Set up checkpointing
    if args.reload_ckpt != "None":
        ckpt = tf.train.Checkpoint(model=model_classifier, global_step=global_step)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_consumed()
    else:
        ckpt = tf.train.Checkpoint(model=model_classifier, global_step=global_step)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3) 

    writer = tf.summary.create_file_writer(manager._directory)
    with writer.as_default(), tf.summary.record_if(lambda: int(global_step.numpy()) % 100 == 0):
        for epoch in range(epoch_start, config.num_epochs):
            
            start_time = time.time()

            random = np.array([0, 1, 2])
            np.random.shuffle(random)
            rand_inputs = [ds_train1, ds_train2, ds_train3]

            train_one_epoch(model_classifier=model_classifier, model_generator= model_generator, 
                model_critic = model_critic, train_input1=rand_inputs[random[0]], 
                train_input2=rand_inputs[random[1]], optimizer1=optimizer1, optimizer2=optimizer2,
                optimizer3=optimizer3, global_step=global_step, config=config)

            train_metr = eval_one_epoch(model_classifier=model_classifier, model_generator=model_generator, dataset=ds_train_complete,
                summary_directory=os.path.join(manager._directory, "train"), 
                global_step=global_step, config=config, training=False)
            
            val_out_metr = eval_one_epoch(model_classifier=model_classifier, model_generator=model_generator, dataset=ds_val_out,
                summary_directory=os.path.join(manager._directory, "val_out"), 
                global_step=global_step, config=config, training=False)

            val_in_metr = eval_one_epoch(model_classifier=model_classifier, model_generator=model_generator, dataset=ds_val_in,
                summary_directory=os.path.join(manager._directory, "val_in"),
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
            logging.info("train_accuracy: {:2f}, train_loss: {:4f}".format(
                train_metr['accuracy'], train_metr['loss']))
            logging.info("val_out_accuracy: {:2f}, val_out_loss: {:4f}".format(
                val_out_metr['accuracy'], val_out_metr['loss']))
            logging.info("val_in_accuracy: {:2f}, val_in_loss: {:4f}".format(
                val_in_metr['accuracy'], val_in_metr['loss']))
           

            if epoch == epoch_start:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                copy_source(dir_path, manager._directory)

    
    # Mark experiment as completed
    # TODO: add other metrics - done
    exp_repo.mark_experiment_as_completed(exp_id, 
        train_accuracy=train_metr['accuracy'],
        val_out_accuracy=val_out_metr['accuracy'],
        val_in_accuracy=val_in_metr['accuracy'])

if __name__ == "__main__":
    main()
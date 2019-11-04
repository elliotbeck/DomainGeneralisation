import csv
import json
from collections import namedtuple
from shutil import make_archive
from datetime import datetime
import os
import tensorflow as tf
import numpy as np

import local_settings

def copy_source(code_directory, model_dir):
    now = datetime.now().strftime('%Y-%m-%d')
    make_archive(os.path.join(model_dir, "code_%s.tar.gz" % now), 'tar', code_directory)



def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj


def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config

def update_config(config, args):
    for entry in config:
        if hasattr(args, entry):
            if eval("args.{}".format(entry)) is not None:
                config[entry] = eval("args.{}".format(entry))
    return config

def compute_optimal_transport(M, r, c, lam=1, epsilon=1e-1):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """


    n, m = tf.shape(M)
    P = tf.math.exp(- lam * M)
    P /= tf.math.reduce_sum(P)
    u = tf.zeros(n)
    # normalize this matrix
    while tf.math.reduce_max(tf.math.abs(u - tf.math.reduce_sum(P, axis=1))) > epsilon:
        u = tf.math.reduce_sum(P, axis=1)
        P *= tf.reshape(r/u,[-1, 1])
        #print(P)
        P *= tf.reshape(c/tf.math.reduce_sum(P, axis=0),[1, -1])
    return P, tf.math.reduce_sum(P * M)
    
    
def compute_cost_matrix(input1, input2):
        norms_true = tf.norm(input1,2)
        norms_generated = tf.norm(input2,2)
        matrix_norms = tf.tensordot(norms_true,norms_generated, axes=0)
        matrix_critic = tf.tensordot(input1,input2, axes=0)
        cost_matrix = 1 - matrix_critic/matrix_norms
        print(cost_matrix)
        return cost_matrix
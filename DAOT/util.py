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

def compute_optimal_transport(M, r, c, lam, epsilon=1e-8):
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


    n, m = M.shape
    P = tf.exp(- lam * M)
    P /= P.sum()
    u = tf.zeros(n)
    # normalize this matrix
    while tf.maximum(tf.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, tf.reduce_sum(P * M)
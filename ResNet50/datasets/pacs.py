import h5py
import json

import tensorflow as tf
from tensorflow_datasets.core import BuilderConfig, dataset_utils
import tensorflow_datasets.public_api as tfds
from PIL import Image

import local_settings
import os
import numpy as np
from absl import flags, app

from random import shuffle

_CITATION = ""

_DESCRIPTION = """\
PCAS
"""

# see https://www.tensorflow.org/datasets/add_dataset

flags.DEFINE_list(name="validation_split", default=["photo"], help="")
flags.DEFINE_string(name="tfds_path", default=None, help="")

flags = flags.FLAGS

VALIDATION_SPLIT = ["photo"]
holdout_domain_path = ["pacs/photo_train.hdf5", "pacs/photo_val.hdf5"]



class PACSConfig(tfds.core.BuilderConfig):
    def __init__(self, validation_split=None, **kwargs):
        self.validation_split = VALIDATION_SPLIT

        if validation_split is not None:
            self.validation_split = validation_split

        super(PACSConfig, self).__init__(
            name="{}".format("_".join(self.validation_split)),
            description="pacs dataset",
            version="0.1.11",
            **kwargs)


class PACS(tfds.core.GeneratorBasedBuilder):
    def __init__(self, validation_split=None, **kwargs):
        config = PACSConfig(validation_split=validation_split)
        super().__init__(config=config, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(
                    shape=(227, 227, 3), #TODO: add as argument and resize accordingly
                    encoding_format="png"),
                "attributes": {
                    "label": tf.int64,
                    "domain": tf.string
                }
            }),
            urls=["http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017"],
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        
        # TODO: remove split defined in validation_split and create separate test set for it
        # TODO: download remaining datasets and fix the filename vars below - done

        # filter hold out domain out from training data
        filenames_train = ['pacs/art_painting_train.hdf5', 'pacs/sketch_train.hdf5',
                     'pacs/cartoon_train.hdf5', 'pacs/photo_train.hdf5']
        train_list = list(set(filenames_train) - set(holdout_domain_path))

        filenames = [train_list[0]]
        train_files1 = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]
        
        filenames = [train_list[1]]
        train_files2 = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = [train_list[2]]
        train_files3 = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = train_list
        train_files_complete = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        # filter hold out domain out from validation data
        filenames_val = ['pacs/art_painting_val.hdf5', 'pacs/sketch_val.hdf5',
                     'pacs/cartoon_val.hdf5', 'pacs/photo_val.hdf5']
        val_list = list(set(filenames_val) - set(holdout_domain_path))

        filenames = val_list
        validation_files_in = [os.path.join(local_settings.RAW_DATA_PATH, f) 
            for f in filenames]

        filenames = [holdout_domain_path[1]]
        validation_files_out = [os.path.join(local_settings.RAW_DATA_PATH, f)
            for f in filenames]

        # filenames = ['pacs/art_painting_test.hdf5', 'pacs/sketch_test.hdf5',
        #              'pacs/cartoon_test.hdf5']
        # test_files = [os.path.join(local_settings.RAW_DATA_PATH, f)
        #               for f in filenames]

        return [tfds.core.SplitGenerator(
                    name="train1",
                    num_shards=30,
                    gen_kwargs=dict(
                        split="train1",
                        files=train_files1
                )),
                tfds.core.SplitGenerator(
                    name="train2",
                    num_shards=30,
                    gen_kwargs=dict(
                        split="train2",
                        files=train_files2
                )),
                tfds.core.SplitGenerator(
                    name="train3",
                    num_shards=30,
                    gen_kwargs=dict(
                        split="train3",
                        files=train_files3
                )),                   
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=1,
                    gen_kwargs=dict(
                        split="train",
                        files=train_files_complete
                )),
                tfds.core.SplitGenerator(
                    name="val_in",
                    num_shards=30,
                    gen_kwargs=dict(
                        split="val_in",
                        files=validation_files_in
                )),
                tfds.core.SplitGenerator(
                    name="val_out",
                    num_shards=1,
                    gen_kwargs=dict(
                        split="val_out",
                        files=validation_files_out))
                # tfds.core.SplitGenerator(
                #     name=tfds.Split.VALIDATION,
                #     num_shards=10,
                #     gen_kwargs=dict(
                #         split="validation",
                #         files=validation_files
                #     )),
                # tfds.core.SplitGenerator(
                #     name=tfds.Split.TEST,
                #     num_shards=1,
                #     gen_kwargs=dict(
                #         split="test",
                #         files=test_files
                # ))
                ]
    

    def _generate_examples(self, split, files):
        
        for f in files:
            file_ = h5py.File(f, 'r')
            images = list(file_['images'])
            labels = list(file_['labels'])

            for img, label_ in zip(images, labels):

                example = {
                    "attributes": {
                        "label": label_,
                        "domain": f.split("_")[0].split("/")[-1]
                    },
                    "image": np.uint8(img)
                }

                yield example




def main(_):
    builder_kwargs = {
        "validation_split": flags.validation_split
    }

    tfdataset_path = local_settings.TF_DATASET_PATH
    if flags.tfds_path is not None:
        tfdataset_path = flags.tfds_path

    train, dsinfo = tfds.load("pacs", 
        data_dir=tfdataset_path, split=tfds.Split.VALIDATION,
        builder_kwargs=builder_kwargs, with_info=True)

    for example in dataset_utils.as_numpy(train):
        import pdb; pdb.set_trace()
        print(example["attributes"]["label"])

if __name__ == '__main__':
    app.run(main)
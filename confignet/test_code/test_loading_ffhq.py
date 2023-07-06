import numpy as np
import argparse
import os
import sys


# setting path
sys.path.append("../")
from confignet.dataset import CustomDataset

real_training_set_path = "/mnt/10TB/FaceDatasets/FFHQ/normalized_train"
output_path = "/mnt/SSD/confignet_stylegan/ConfigNet/confignet/data/ffhq_train.pckl"
# initilize the dataset
dataset = CustomDataset((256, 256, 3), False)
img_output_dir = None

dataset.generate_face_dataset(
    real_training_set_path,
    output_path,
    attribute_label_file_path=None,
    pre_normalize=False,
)
if img_output_dir is not None:
    print("Writing aligned images to %s" % (img_output_dir))
    dataset.write_images(img_output_dir)

# real_training_set = CustomDataset.load(real_training_set_path)

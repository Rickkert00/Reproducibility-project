import json
import os
from copy import copy

import tensorflow as tf

def copy_sample(sample_size, set_size, set_folder, sample_folder):
    step = set_size / sample_size
    total_transforms_dict = json.load(open(os.path.join(set_folder, "../total_transforms_train.json"), "r"))
    current_transforms_dict = copy(total_transforms_dict)
    # clear the list of the current transforms dict so we can start updating it
    current_transforms_dict["frames"] = []

    # clear the training folder
    for filename in os.listdir(sample_folder):
        os.remove(os.path.join(sample_folder, filename))

    # spaced evenly throughout the set we sample and store in the training folder
    for i in range(sample_size):
        current_index = round(step * i)
        current_name = total_transforms_dict["frames"][current_index]["file_path"].split("/")[-1]
        img = tf.io.read_file(os.path.join(set_folder, f"{current_name}.png"))
        current_transforms_dict["frames"].append(total_transforms_dict["frames"][current_index])
        tf.io.write_file(os.path.join(sample_folder, f"{current_name}.png"), img)

    # update the transforms dictionary accordingly
    with open(os.path.join(sample_folder, "../transforms_train.json"), "w") as file_writer:
        json.dump(current_transforms_dict, file_writer, indent=4)


def split_training_test(training_size, set_folder, train_folder, test_folder):
    set_size = len(os.listdir(set_folder)) // 2  # divide by two because there are mask in test folder too
    step_size = set_size / training_size
    training_indexes = set()
    for i in range(training_size):
        current_index = round(step_size * i)
        training_indexes.add(current_index)

    total_transforms = json.load(open(os.path.join(set_folder, "../transforms_test.json"), "r"))
    train_transforms = copy(total_transforms)
    test_transforms = copy(total_transforms)
    train_transforms["frames"] = []
    test_transforms["frames"] = []
    for i in range(set_size):
        img = tf.io.read_file(os.path.join(set_folder, f"r_{i}.png"))
        if i in training_indexes:
            tf.io.write_file(os.path.join(train_folder, f"r_{i}.png"), img)
            train_transforms["frames"].append(total_transforms["frames"][i])
        else:
            mask_img = tf.io.read_file(os.path.join(set_folder, f"r_{i}_depth_0001.png"))
            tf.io.write_file(os.path.join(test_folder, f"r_{i}.png"), img)
            tf.io.write_file(os.path.join(test_folder, f"r_{i}_depth_0001.png"), mask_img)
            test_transforms["frames"].append(total_transforms["frames"][i])

    with open(os.path.join(train_folder, "../transforms_train.json"), "w") as file_writer:
        json.dump(train_transforms, file_writer, indent=4)

    with open(os.path.join(test_folder, "../transforms_test.json"), "w") as file_writer:
        json.dump(test_transforms, file_writer, indent=4)

if __name__ == '__main__':
    # copy sample
    SAMPLE_SIZE = 60
    SET_SIZE = 120
    assert SAMPLE_SIZE <= SET_SIZE

    set_folder = "../data/nerf_raw_noise/lego/total_train/"
    sample_folder = "../data/nerf_raw_noise/lego/train/"

    copy_sample(SAMPLE_SIZE, SET_SIZE, set_folder, sample_folder)

    # # split the test set into training and test set
    # training_size = 120
    # set_folder = "../data/nerf_synthetic/lego/test/"
    # train_folder = "../data/our_nerf_synthetic/lego/train/"
    # test_folder = "../data/our_nerf_synthetic/lego/test/"
    #
    # split_training_test(training_size, set_folder, train_folder, test_folder)


# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# - Load models
from model.rfn_unet_v4_wrapper import RefinedUNetV4

# - Load model config
from model.rfn_unet_v4_config import RefinedUNetV4Config

# - Load inference config
from config.inference_config import InferenceConfig

# - Load utils
from util.tfrecordloader_l8 import load_testset
from util.tile_utils import reconstruct_full

# from util.tfrecord_util import load_tfrecord
# from util.reconstruct_util import reconstruct

# # - Path to input and output
# data_path = "../../data/l8/testcase/"
# save_path = "output"

# # - Critical hyper-parameters
# theta_alpha, theta_beta = 80, 0.03125
# theta_gamma = 3

# # - Hyper-parameters
# CROP_HEIGHT, CROP_WIDTH = 512, 512
# d_bifeats, d_spfeats = 5, 2
# num_bands = 7
# num_classes = 4
# batch_size = 1
# ugenerator_path = "parameter/saved_model/unary_generator"
# crf_computation_path = "parameter/saved_model/crf_computation"


def main():
    # - Instantiate inference config
    inference_config = InferenceConfig()

    # - Output all attributes of the inference config
    print(inference_config.__dict__)

    # - Create output dir
    if not os.path.exists(inference_config.save_path):
        os.makedirs(inference_config.save_path)
        print("Create {}.".format(inference_config.save_path))

    # - Get all names of data files
    print("Load from {}".format(inference_config.data_path))
    test_names = os.listdir(inference_config.data_path)
    print("Data list: {}".format(test_names))

    # - Instantiate model
    model_config = RefinedUNetV4Config(
        height=inference_config.crop_height,
        width=inference_config.crop_width,
        n_channels=inference_config.num_bands,
        num_classes=inference_config.num_classes,
        theta_alpha=inference_config.theta_alpha,
        theta_beta=inference_config.theta_beta,
        theta_gamma=inference_config.theta_gamma,
        bilateral_compat=inference_config.bilateral_compat,
        spatial_compat=inference_config.spatial_compat,
        num_iterations=inference_config.num_iterations,
    )
    model = RefinedUNetV4(
        config=model_config,
        ugenerator_path=inference_config.ugenerator_path,
    )

    with open(
        os.path.join(inference_config.save_path, inference_config.save_info_fname), "w"
    ) as fp:
        fp.writelines("name, theta_alpha, theta_beta, theta_gamma, duration\n")

        for test_name in test_names:
            # Names
            save_npz_name = test_name.replace("train.tfrecords", "rfn.npz")
            save_png_name = test_name.replace("train.tfrecords", "rfn.png")

            # Load one test case
            test_name = [os.path.join(inference_config.data_path, test_name)]
            test_set = load_testset(
                test_name,
                batch_size=1,
            )
            refinements = []

            # Inference
            start = time.time()
            i = 0
            for record in test_set.take(-1):
                print("Patch {}...".format(i))
                x = record["x_train"]
                rfn = model.inference(x[0])

                refinements += [rfn]

                i += 1

            refinements = np.stack(refinements, axis=0)
            refinement = reconstruct_full(
                refinements,
                crop_height=inference_config.crop_height,
                crop_width=inference_config.crop_width,
            )
            duration = time.time() - start

            # Save
            np.savez(
                os.path.join(inference_config.save_path, save_npz_name), refinement
            )
            print(
                "Write to {}".format(
                    os.path.join(inference_config.save_path, save_npz_name)
                )
            )
            plt.imsave(
                os.path.join(inference_config.save_path, save_png_name), refinement
            )
            print(
                "Write to {}".format(
                    os.path.join(inference_config.save_path, save_png_name)
                )
            )

            fp.writelines(
                "{}, {}, {}, {}, {}\n".format(
                    test_name,
                    inference_config.theta_alpha,
                    inference_config.theta_beta,
                    inference_config.theta_gamma,
                    duration,
                )
            )
            fp.flush()

            print("{} Done.".format(test_name))


if __name__ == "__main__":
    main()

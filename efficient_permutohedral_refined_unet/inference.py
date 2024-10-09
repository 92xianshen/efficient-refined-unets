# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# - Load models
from model.perm_rfn_unet_x_wrapper import PermutohedralRefinedUNetX

# - Load model config
from model.perm_rfn_unet_x_config import PermRfnUNetXConfig

# - Load inference config
from config.inference_config import InferenceConfig

# - Load utils
from util.tfrecordloader_l8 import load_testset
from util.tile_utils import reconstruct_full


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
    model_config = PermRfnUNetXConfig(
        height=inference_config.crop_height,
        width=inference_config.crop_width,
        n_channels=inference_config.num_bands,
        num_classes=inference_config.num_classes,
        d_bifeats=inference_config.d_bifeats,
        d_spfeats=inference_config.d_spfeats,
        theta_alpha=inference_config.theta_alpha,
        theta_beta=inference_config.theta_beta,
        theta_gamma=inference_config.theta_gamma,
        bilateral_compat=inference_config.bilateral_compat,
        spatial_compat=inference_config.spatial_compat,
        num_iterations=inference_config.num_iterations,
    )
    model = PermutohedralRefinedUNetX(
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

"""
Baseline class of Permutohedral Refined UNet X. 
"""
import tensorflow as tf

from .perm_rfn_unet_x_config import PermRfnUNetXConfig
from .perm_crf_computation import CRFComputation
from .linear2 import Linear2


class PermutohedralRefinedUNetX(tf.Module):
    def __init__(
        self,
        config: PermRfnUNetXConfig = None,
        ugenerator_path: str = None,
        name: str = None,
    ):
        super().__init__(name)
        self.config = config
        self.crf_computation = CRFComputation(self.config)
        self.unary_generator = tf.saved_model.load(ugenerator_path)

        self.linear2 = Linear2()

        print("Activate modules...")
        self.unary_generator(
            tf.ones(
                shape=[
                    1,
                    self.config.height,
                    self.config.width,
                    self.config.n_channels,
                ],
                dtype=tf.float32,
            )
        )

        self.linear2(
            tf.ones(
                shape=[
                    self.config.height,
                    self.config.width,
                    self.config.d_bifeats - 2,
                ],
                dtype=tf.float32,
            )
        )

        (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        ) = self.crf_computation.init_partially(
            tf.ones(
                shape=[
                    self.config.height,
                    self.config.width,
                    self.config.d_bifeats - 2,
                ],
                dtype=tf.float32,
            )
        )

        bilateral_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                bilateral_coords_1d_uniq,
                tf.range(self.crf_computation.bilateral_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        bilateral_blur_neighbors = bilateral_hash_table.lookup(bilateral_ns) + 1

        spatial_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                spatial_coords_1d_uniq,
                tf.range(self.crf_computation.spatial_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        spatial_blur_neighbors = spatial_hash_table.lookup(spatial_ns) + 1

        self.crf_computation.mean_field_approximation(
            tf.ones(
                shape=[config.height, config.width, config.num_classes],
                dtype=tf.float32,
            ),
            bilateral_blur_neighbors=bilateral_blur_neighbors,
            spatial_blur_neighbors=spatial_blur_neighbors,
        )

        print("Modules activated.")

    def inference(self, image: tf.Tensor) -> tf.Tensor:
        """
        Predict and refine.

        Args:
            image: 7-band input, [h, w, c], float32.

        Returns:
            rfn: [h, w],
        """

        unary = self.unary_generator(image[tf.newaxis, ...])[0]  # [H, W, N]
        ref = self.linear2(image[..., 4:1:-1])  # [H, W, 3]

        (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        ) = self.crf_computation.init_partially(ref)

        bilateral_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                bilateral_coords_1d_uniq,
                tf.range(self.crf_computation.bilateral_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        bilateral_blur_neighbors = bilateral_hash_table.lookup(bilateral_ns) + 1

        spatial_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                spatial_coords_1d_uniq,
                tf.range(self.crf_computation.spatial_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        spatial_blur_neighbors = spatial_hash_table.lookup(spatial_ns) + 1

        rfn = self.crf_computation.mean_field_approximation(
            unary,
            bilateral_blur_neighbors=bilateral_blur_neighbors,
            spatial_blur_neighbors=spatial_blur_neighbors,
        )

        return rfn

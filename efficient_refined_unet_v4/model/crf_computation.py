import tensorflow as tf

from .bilateral_filter_computation import BilateralHighDimFilterComputation
from .spatial_filter_computation import SpatialHighDimFilterComputation


class CRFComputation(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of `unary`, [H, W, N], float32
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), # of `image`, [H, W, 3], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_alpha`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_beta`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_gamma`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `bilateral_compat`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `spatial_compat`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.float32), # of `compatibility`, [], float32
        tf.TensorSpec(shape=[], dtype=tf.int32), # of `num_iterations`, [], int32
    ])
    def mean_field_approximation(
        self,
        unary: tf.Tensor,
        image: tf.Tensor,
        theta_alpha: tf.float32,
        theta_beta: tf.float32,
        theta_gamma: tf.float32,
        bilateral_compat: tf.float32,
        spatial_compat: tf.float32,
        compatibility: tf.float32,
        num_iterations: tf.int32,
    ) -> tf.Tensor:
        unary_shape = tf.shape(unary)
        height, width, num_classes = unary_shape[0], unary_shape[1], unary_shape[2]

        # Non-critical parameter
        bilateral_range_padding = 2
        bilateral_space_padding = 2
        spatial_space_padding = 2
        n_iters = 2  # conv iteration in the filters

        # - Initialize
        bilateral_filter = BilateralHighDimFilterComputation()
        spatial_filter = SpatialHighDimFilterComputation()

        (
            bilateral_splat_coords,
            bilateral_data_size,
            bilateral_data_shape,
            bilateral_slice_idx,
            bilateral_alpha_prod,
        ) = bilateral_filter.init(
            image,
            range_sigma=theta_beta,
            space_sigma=theta_alpha,
            range_padding=bilateral_range_padding,
            space_padding=bilateral_space_padding,
        )

        (
            spatial_splat_coords,
            spatial_data_size,
            spatial_data_shape,
            spatial_slice_idx,
            spatial_alpha_prod,
        ) = spatial_filter.init(
            height, width, space_sigma=theta_gamma, space_padding=spatial_space_padding
        )

        # - Compute symmetric weights
        all_ones = tf.ones([height, width, 1], dtype=tf.float32)  # [H, W, 1]
        bilateral_norm_vals = bilateral_filter.compute(
            all_ones,
            splat_coords=bilateral_splat_coords,
            data_size=bilateral_data_size,
            data_shape=bilateral_data_shape,
            slice_idx=bilateral_slice_idx,
            alpha_prod=bilateral_alpha_prod,
            n_iters=n_iters,
        )
        bilateral_norm_vals = 1.0 / (bilateral_norm_vals**0.5 + 1e-20)
        spatial_norm_vals = spatial_filter.compute(
            all_ones,
            splat_coords=spatial_splat_coords,
            data_size=spatial_data_size,
            data_shape=spatial_data_shape,
            slice_idx=spatial_slice_idx,
            alpha_prod=spatial_alpha_prod,
            n_iters=n_iters,
        )
        spatial_norm_vals = 1.0 / (spatial_norm_vals**0.5 + 1e-20)

        # - Initialize Q
        Q = tf.nn.softmax(-unary, axis=-1)  # [H, W, N]

        for i in range(num_iterations):
            tmp1 = -unary  # [H, W, N]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = bilateral_filter.compute(
                Q * bilateral_norm_vals,
                splat_coords=bilateral_splat_coords,
                data_size=bilateral_data_size,
                data_shape=bilateral_data_shape,
                slice_idx=bilateral_slice_idx,
                alpha_prod=bilateral_alpha_prod,
                n_iters=n_iters,
            )
            bilateral_out *= bilateral_norm_vals

            # - Symmetric normalization and spatial message passing
            spatial_out = spatial_filter.compute(
                Q * spatial_norm_vals,
                splat_coords=spatial_splat_coords,
                data_size=spatial_data_size,
                data_shape=spatial_data_shape,
                slice_idx=spatial_slice_idx,
                alpha_prod=spatial_alpha_prod,
                n_iters=n_iters,
            )
            spatial_out *= spatial_norm_vals

            # - Message passing
            message_passing = (
                bilateral_compat * bilateral_out + spatial_compat * spatial_out
            )  # [H, W, C]

            # - Compatibility transform
            pairwise = compatibility * message_passing  # [N, C]

            # - Local update
            tmp1 -= pairwise  # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1)  # [N, C]

        # - Maximum posterior estimation
        MAP = tf.math.argmax(Q, axis=-1)  # [H, W]

        return MAP

"""
`CRFComputation` includes all static computations. 
Should be built once before use.
"""

import tensorflow as tf
from .perm_rfn_unet_x_config import PermRfnUNetXConfig

from .permutohedralx_variable import PermutohedralXVariable
from .permutohedralx_function import PermutohedralXFunction


class CRFComputation(tf.Module):
    def __init__(self, config: PermRfnUNetXConfig, name=None):
        super().__init__(name)

        self.config = config

        self.bilateral_computation = PermutohedralXFunction(
            N=self.config.n_feats, d=self.config.d_bifeats
        )
        self.spatial_computation = PermutohedralXFunction(
            N=self.config.n_feats, d=self.config.d_spfeats
        )

        self.bilateral_vars = PermutohedralXVariable(
            N=self.config.n_feats, d=self.config.d_bifeats
        )
        self.spatial_vars = PermutohedralXVariable(
            N=self.config.n_feats, d=self.config.d_spfeats
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def create_feature(self, image: tf.Tensor) -> tf.Tensor:
        # - Create bilateral features.
        ys, xs = tf.meshgrid(
            tf.range(self.config.height), tf.range(self.config.width), indexing="ij"
        )  # [h, w]
        ys_bifeats, xs_bifeats = (
            tf.cast(ys, dtype=tf.float32) / self.config.theta_alpha,
            tf.cast(xs, dtype=tf.float32) / self.config.theta_alpha,
        )
        color_feats = image / self.config.theta_beta
        bilateral_feat = tf.concat(
            [xs_bifeats[..., tf.newaxis], ys_bifeats[..., tf.newaxis], color_feats],
            axis=-1,
        )
        bilateral_feat = tf.reshape(bilateral_feat, shape=[-1, self.config.d_bifeats])

        # - Create spatial features.
        ys_spfeats, xs_spfeats = (
            tf.cast(ys, dtype=tf.float32) / self.config.theta_gamma,
            tf.cast(xs, dtype=tf.float32) / self.config.theta_gamma,
        )
        spatial_feat = tf.concat(
            [xs_spfeats[..., tf.newaxis], ys_spfeats[..., tf.newaxis]], axis=-1
        )
        spatial_feat = tf.reshape(spatial_feat, shape=[-1, self.config.d_spfeats])

        return bilateral_feat, spatial_feat

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def init_partially(self, image: tf.Tensor) -> None:
        bilateral_feat, spatial_feat = self.create_feature(image)

        (
            bilateral_os,
            bilateral_ws,
            bilateral_coords_1d_uniq,
            bilateral_M,
            bilateral_ns,
        ) = self.bilateral_computation.init(bilateral_feat)
        self.bilateral_vars.os.assign(bilateral_os)
        self.bilateral_vars.ws.assign(bilateral_ws)
        self.bilateral_vars.M.assign(bilateral_M)

        (
            spatial_os,
            spatial_ws,
            spatial_coords_1d_uniq,
            spatial_M,
            spatial_ns,
        ) = self.spatial_computation.init(spatial_feat)
        self.spatial_vars.os.assign(spatial_os)
        self.spatial_vars.ws.assign(spatial_ws)
        self.spatial_vars.M.assign(spatial_M)

        return (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, None], dtype=tf.float32
            ),  # of `unary`, [H, W, N], float32
            tf.TensorSpec(
                shape=[2, None, None], dtype=tf.int32
            ),  # of bilateral_blur_neighbors, [2, M, d + 1], int32
            tf.TensorSpec(
                shape=[2, None, None], dtype=tf.int32
            ),  # of spatial_blur_neighbors, [2, M, d + 1], int32
        ]
    )
    def mean_field_approximation(
        self,
        unary: tf.Tensor,
        bilateral_blur_neighbors: tf.Tensor,
        spatial_blur_neighbors: tf.Tensor,
    ) -> tf.Tensor:
        unary_shape = tf.shape(unary)  # [H, W, C] kept
        # - Compute symmetric weights
        all_one = tf.ones(shape=[self.config.n_feats, 1], dtype=tf.float32)  # [N, 1]
        bilateral_norm_val = self.bilateral_computation.compute(
            all_one,
            self.bilateral_vars.os,
            self.bilateral_vars.ws,
            bilateral_blur_neighbors,
            self.bilateral_vars.M,
        )
        bilateral_norm_val = 1.0 / (bilateral_norm_val**0.5 + 1e-20)
        spatial_norm_val = self.spatial_computation.compute(
            all_one,
            self.spatial_vars.os,
            self.spatial_vars.ws,
            spatial_blur_neighbors,
            self.spatial_vars.M,
        )
        spatial_norm_val = 1.0 / (spatial_norm_val**0.5 + 1e-20)

        # - Initialize Q
        unary = tf.reshape(
            unary, shape=[self.config.n_feats, self.config.num_classes]
        )  # flatten, [N, C]
        Q = tf.nn.softmax(-unary, axis=-1)  # [N, C]

        for _ in range(self.config.num_iterations):
            tmp1 = -unary  # [N, C]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = self.bilateral_computation.compute(
                Q * bilateral_norm_val,
                self.bilateral_vars.os,
                self.bilateral_vars.ws,
                bilateral_blur_neighbors,
                self.bilateral_vars.M,
            )  # [N, C]
            bilateral_out *= bilateral_norm_val  # [N, C]

            # - Symmetric normalization and spatial message passing
            spatial_out = self.spatial_computation.compute(
                Q * spatial_norm_val,
                self.spatial_vars.os,
                self.spatial_vars.ws,
                spatial_blur_neighbors,
                self.spatial_vars.M,
            )  # [N, C]
            spatial_out *= spatial_norm_val  # [N, C]

            # - Message passing
            message_passing = (
                self.config.bilateral_compat * bilateral_out
                + self.config.spatial_compat * spatial_out
            )  # [N, C]

            # - Compatibility transform
            pairwise = self.config.compatibility * message_passing  # [N, C]

            # - Local update
            tmp1 -= pairwise  # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1)  # [N, C]

        # - Maximum posterior
        Q = tf.reshape(Q, shape=unary_shape)  # [H, W, C]
        MAP = tf.math.argmax(Q, axis=-1)  # [H, W]

        return MAP

"""
Bilateral high-dim filter, built with TF 2.x.
"""

import tensorflow as tf


class BilateralHighDimFilterComputation(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, 3], dtype=tf.float32
            ),  # of `features`, [H, W, C], float32
            tf.TensorSpec(shape=[], dtype=tf.float32),  # of `range_sigma`, [], float32
            tf.TensorSpec(shape=[], dtype=tf.float32),  # of `space_sigma`, [], float32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of `range_padding`, [], int32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of `space_padding`, [], int32
        ]
    )
    def init(
        self,
        image: tf.Tensor,
        range_sigma: tf.float32,
        space_sigma: tf.float32,
        range_padding: tf.int32,
        space_padding: tf.int32,
    ) -> tf.Tensor:
        """
        Initialize, `feature` should be three-channel and channel-last
        """

        def clamp(
            x: tf.Tensor, min_value: tf.float32, max_value: tf.float32
        ) -> tf.Tensor:
            return tf.maximum(min_value, tf.minimum(max_value, x))

        def get_both_indices(coord: tf.Tensor, size: tf.int32) -> tf.Tensor:
            """
            Method to get left and right indices of slice interpolation.
            """
            left_index = clamp(
                tf.cast(coord, dtype=tf.int32), min_value=0, max_value=size - 1
            )
            right_index = clamp(left_index + 1, min_value=0, max_value=size - 1)
            return left_index, right_index

        # Decompose `features` into r, g, and b channels
        def create_range_params(ch):
            ch_min, ch_max = tf.reduce_min(ch), tf.reduce_max(ch)
            ch_delta = ch_max - ch_min
            # - Range coordinates, shape [H, W], dtype float
            chh = ch - ch_min

            # - Depths of data grid
            small_chdepth = (
                tf.cast(ch_delta / range_sigma, dtype=tf.int32) + 1 + 2 * range_padding
            )

            # - Range coordinates of splat, shape [H, W]
            splat_chh = tf.cast(chh / range_sigma + 0.5, dtype=tf.int32) + range_padding

            # - Range coordinates of slice, shape [H, W]
            slice_chh = chh / range_sigma + tf.cast(range_padding, dtype=tf.float32)

            # - Slice interpolation range coordinate pairs
            ch_index, chh_index = get_both_indices(
                slice_chh, small_chdepth
            )  # [H, W], [H, W]

            # - Intepolation factors
            ch_alpha = tf.reshape(
                slice_chh - tf.cast(ch_index, dtype=tf.float32),
                shape=[
                    -1,
                ],
            )  # [H x W, ]

            return small_chdepth, splat_chh, ch_index, chh_index, ch_alpha

        height, width = tf.shape(image)[0], tf.shape(image)[1]
        size = height * width
        dim = 5

        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        small_rdepth, splat_rr, r_index, rr_index, r_alpha = create_range_params(r)
        small_gdepth, splat_gg, g_index, gg_index, g_alpha = create_range_params(g)
        small_bdepth, splat_bb, b_index, bb_index, b_alpha = create_range_params(b)

        # Height and width of data grid, scala, dtype int
        small_height = (
            tf.cast(tf.cast(height - 1, tf.float32) / space_sigma, dtype=tf.int32)
            + 1
            + 2 * space_padding
        )
        small_width = (
            tf.cast(tf.cast(width - 1, tf.float32) / space_sigma, dtype=tf.int32)
            + 1
            + 2 * space_padding
        )

        # Space coordinates, shape [H, W], dtype int
        yy, xx = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")  # [H, W]
        yy, xx = tf.cast(yy, dtype=tf.float32), tf.cast(xx, dtype=tf.float32)
        # Spatial coordinates of splat, shape [H, W]
        splat_yy = tf.cast(yy / space_sigma + 0.5, dtype=tf.int32) + space_padding
        splat_xx = tf.cast(xx / space_sigma + 0.5, dtype=tf.int32) + space_padding
        # Spatial coordinates of slice, shape [H, W]
        slice_yy = tf.cast(yy, dtype=tf.float32) / space_sigma + tf.cast(
            space_padding, dtype=tf.float32
        )
        slice_xx = tf.cast(xx, dtype=tf.float32) / space_sigma + tf.cast(
            space_padding, dtype=tf.float32
        )

        # Spatial interpolation index of slice
        y_index, yy_index = get_both_indices(slice_yy, small_height)  # [H, W]
        x_index, xx_index = get_both_indices(slice_xx, small_width)  # [H, W]

        # Spatial interpolation factor of slice
        y_alpha = tf.reshape(
            slice_yy - tf.cast(y_index, dtype=tf.float32),
            shape=[
                -1,
            ],
        )  # [H x W, ]
        x_alpha = tf.reshape(
            slice_xx - tf.cast(x_index, dtype=tf.float32),
            shape=[
                -1,
            ],
        )  # [H x W, ]

        # - Bilateral interpolation index and factor
        interp_indices = [
            y_index,
            yy_index,
            x_index,
            xx_index,
            r_index,
            rr_index,
            g_index,
            gg_index,
            b_index,
            bb_index,
        ]  # [10, H x W]
        alphas = [
            1.0 - y_alpha,
            y_alpha,
            1.0 - x_alpha,
            x_alpha,
            1.0 - r_alpha,
            r_alpha,
            1.0 - g_alpha,
            g_alpha,
            1.0 - b_alpha,
            b_alpha,
        ]  # [10, H x W]

        # Method of coordinate transformation
        def coord_transform(idx):
            return tf.reshape(
                (
                    (
                        (idx[:, 0, :] * small_width + idx[:, 1, :]) * small_rdepth
                        + idx[:, 2, :]
                    )
                    * small_gdepth
                    + idx[:, 3, :]
                )
                * small_bdepth
                + idx[:, 4, :],
                shape=[
                    -1,
                ],
            )  # [2^d x H x W, ]

        # Initialize interpolation
        offset = tf.range(dim) * 2  # [d, ]
        # Permutation
        permutations = tf.stack(
            tf.meshgrid(
                tf.range(2),
                tf.range(2),
                tf.range(2),
                tf.range(2),
                tf.range(2),
                indexing="ij",
            ),
            axis=-1,
        )
        permutations = tf.reshape(permutations, shape=[-1, dim])  # [2^d, d]
        permutations += offset[tf.newaxis, ...]
        permutations = tf.reshape(
            permutations,
            shape=[
                -1,
            ],
        )  # flatten, [2^d x d, ]
        alpha_prods = tf.reshape(
            tf.gather(alphas, permutations), shape=[-1, dim, size]
        )  # [2^d, d, H x W]
        idx = tf.reshape(
            tf.gather(interp_indices, permutations), shape=[-1, dim, size]
        )  # [2^d, d, H x W]

        # Shape and size of bialteral data grid
        data_shape = tf.stack(
            [
                small_height,
                small_width,
                small_rdepth,
                small_gdepth,
                small_bdepth,
            ],
        )
        data_size = (
            small_height * small_width * small_rdepth * small_gdepth * small_bdepth
        )

        # Bilateral splat coordinates, shape [H x W, ]
        splat_coords = (
            ((splat_yy * small_width + splat_xx) * small_rdepth + splat_rr)
            * small_gdepth
            + splat_gg
        ) * small_bdepth + splat_bb
        splat_coords = tf.reshape(
            splat_coords,
            shape=[
                -1,
            ],
        )  # [H x W, ]

        # Interpolation indices and alphas of bilateral slice
        slice_idx = coord_transform(idx)
        alpha_prod = tf.math.reduce_prod(alpha_prods, axis=1)  # [2^d, H x W]

        return splat_coords, data_size, data_shape, slice_idx, alpha_prod

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, None], dtype=tf.float32
            ),  # of `inp`, [H, W, N],
            tf.TensorSpec(
                shape=[
                    None,
                ],
                dtype=tf.int32,
            ),  # of `splat_coords`, [H x W, ], int32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of `data_size`, [], int32
            tf.TensorSpec(
                shape=[
                    None,
                ],
                dtype=tf.int32,
            ),  # of `data_shape`, [d, ], int32
            tf.TensorSpec(
                shape=[
                    None,
                ],
                dtype=tf.int32,
            ),  # of `slice_idx`, [2^d x H x W, ], int32
            tf.TensorSpec(
                shape=[
                    None,
                    None,
                ],
                dtype=tf.float32,
            ),  # of `alpha_prod`, [H x W, ], float32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of `n_iters`, [], int32
        ]
    )
    def compute(
        self,
        inp: tf.Tensor,
        splat_coords: tf.Tensor,
        data_size: tf.int32,
        data_shape: tf.Tensor,
        slice_idx: tf.Tensor,
        alpha_prod: tf.Tensor,
        n_iters: tf.int32,
    ) -> tf.Tensor:
        height, width = tf.shape(inp)[0], tf.shape(inp)[1]
        size = height * width
        dim = tf.constant(5, dtype=tf.int32)

        # Channel-last to channel-first because tf.map_fn
        inpT = tf.transpose(inp, perm=[2, 0, 1])

        def ch_filter(inp_ch: tf.Tensor) -> tf.Tensor:
            # Filter each channel
            inp_flat = tf.reshape(
                inp_ch,
                shape=[
                    -1,
                ],
            )  # [H x W, ]
            # ==== Splat ====
            data_flat = tf.math.bincount(
                splat_coords,
                weights=inp_flat,
                minlength=data_size,
                maxlength=data_size,
                dtype=tf.float32,
            )
            data = tf.reshape(data_flat, shape=data_shape)

            # ==== Blur ====
            buffer = tf.zeros_like(data)
            perm = [1, 2, 3, 4, 0]

            for _ in range(n_iters):
                buffer, data = data, buffer

                for _ in range(dim):
                    newdata = (buffer[:-2] + buffer[2:]) / 2.0
                    data = tf.concat([data[:1], newdata, data[-1:]], axis=0)
                    data = tf.transpose(data, perm=perm)
                    buffer = tf.transpose(buffer, perm=perm)

            del buffer

            # ==== Slice ====
            data_slice = tf.gather(
                tf.reshape(
                    data,
                    shape=[
                        -1,
                    ],
                ),
                slice_idx,
            )  # (2^dim x h x w)
            data_slice = tf.reshape(data_slice, shape=[-1, size])  # (2^dim, h x w)
            interpolations = alpha_prod * data_slice
            interpolation = tf.reduce_sum(interpolations, axis=0)
            interpolation = tf.reshape(interpolation, shape=[height, width])

            return interpolation

        outT = tf.map_fn(ch_filter, inpT)
        out = tf.transpose(outT, perm=[1, 2, 0])  # Channel-first to channel-last

        return out

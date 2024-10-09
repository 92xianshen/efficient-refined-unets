# -*- coding: utf-8 -*-

"""
- TF implementation of permutohedral lattice, channel-last as well.
- `tf.float32` and `tf.int32` as default float and integer types, respectively.
- Refine the source code
- Update of v3: refine the key size, d + 1 ->> d

- 2023.07.04: Use `tf.lookup.StaticHashTable`.
- 2023.07.14: Remove `tf.lookup.StaticHashTable`, external look-up.
+ 2023.07.19: Remove `N` of `__init__()`.
+ 2023.07.20: Remove comments.
+ 2023.07.20: Partition constants and computation.
+ 2023.09.18: Update `coords_1d`
"""

import numpy as np
import tensorflow as tf


class PermutohedralXFunction(tf.Module):
    def __init__(self, N: tf.int32, d: tf.int32, name=None):
        super().__init__(name)

        self.N = tf.constant(N, dtype=tf.int32)
        self.d = tf.constant(d, dtype=tf.int32)  # size and dimension

        canonical = np.zeros((d + 1, d + 1), dtype=np.int32)  # (d + 1, d + 1)
        for i in range(d + 1):
            canonical[i, : d + 1 - i] = i
            canonical[i, d + 1 - i :] = i - (d + 1)
        self.canonical = tf.constant(canonical, dtype=tf.int32)  # [d + 1, d + 1]

        E = np.vstack(
            [
                np.ones((d,), dtype=np.float32),
                np.diag(-np.arange(d, dtype=np.float32) - 2)
                + np.triu(np.ones((d, d), dtype=np.float32)),
            ]
        )  # (d + 1, d)
        self.E = tf.constant(E, dtype=tf.float32)  # [d + 1, d]

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2.0 / 3.0) * np.float32(d + 1)

        # Compute the diagonal part of E (p.5 in [Adams et al 2010])
        scale_factor = (
            1.0 / np.sqrt((np.arange(d) + 2) * (np.arange(d) + 1)) * inv_std_dev
        )  # (d, )
        self.scale_factor = tf.constant(scale_factor, dtype=tf.float32)  # [d, ]

        diff_valid = 1 - np.tril(
            np.ones((d + 1, d + 1), dtype=np.int32)
        )  # [d + 1, d + 1]
        self.diff_valid = tf.constant(diff_valid, dtype=tf.int32)  # [d + 1, d + 1]

        # Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
        self.alpha = tf.constant(
            1.0 / (1.0 + tf.pow(2.0, -tf.cast(d, dtype=tf.float32))), dtype=tf.float32
        )

        # Helper constant values (matrices).
        d_mat = np.ones((d + 1,), dtype=np.short) * d  # [d + 1, ]
        d_mat = np.diag(d_mat)  # [d + 1, d + 1]
        diagone = np.diag(np.ones(d + 1, dtype=np.short))  # [d + 1, d + 1]
        self.d_mat = tf.constant(d_mat, dtype=tf.int32)  # [d + 1, d + 1]
        self.diagone = tf.constant(diagone, dtype=tf.int32)  # [d + 1, d + 1]

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None], dtype=tf.float32
            ),  # of feature, [N, d], float32
        ]
    )
    def init(self, feature: tf.Tensor) -> tf.Tensor:
        # - Compute the simplex each feature lies in
        # - !!! Shape of feature [N, d]
        # - Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
        cf = feature * self.scale_factor[tf.newaxis, ...]  # [N, d]
        elevated = tf.matmul(cf, tf.transpose(self.E, perm=[1, 0]))  # [N, d + 1]

        # - Find the closest 0-colored simplex through rounding
        down_factor = 1.0 / tf.cast(self.d + 1, dtype=tf.float32)
        up_factor = tf.cast(self.d + 1, dtype=tf.float32)
        v = down_factor * elevated  # [N, d + 1]
        up = tf.math.ceil(v) * up_factor  # [N, d + 1]
        down = tf.math.floor(v) * up_factor  # [N, d + 1]
        rem0 = tf.cast(
            tf.where(up - elevated < elevated - down, up, down), dtype=tf.float32
        )  # [N, d + 1]
        _sum = tf.cast(
            tf.reduce_sum(rem0, axis=1) * down_factor, dtype=tf.int32
        )  # [N, ]

        # - Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        rank = tf.zeros(shape=[self.N, self.d + 1], dtype=tf.int32)  # [N, d + 1]
        diff = elevated - rem0  # [N, d + 1]
        diff_i = diff[..., tf.newaxis]  # [N, d + 1, 1]
        diff_j = diff[..., tf.newaxis, :]  # [N, 1, d + 1]
        di_lt_dj = tf.where(diff_i < diff_j, 1, 0)  # [N, d + 1, d + 1]
        di_geq_dj = tf.where(diff_i >= diff_j, 1, 0)  # [N, d + 1, d + 1]
        rank = rank + tf.reduce_sum(
            di_lt_dj * self.diff_valid[tf.newaxis, ...], axis=2
        )  # [N, d + 1]
        rank = rank + tf.reduce_sum(
            di_geq_dj * self.diff_valid[tf.newaxis, ...], axis=1
        )  # [N, d + 1]

        # - If the point doesn't lie on the plane (sum != 0) bring it back
        rank = rank + _sum[..., tf.newaxis]  # [N, d + 1]
        ls_zero = rank < 0  # [N, d + 1]
        gt_d = rank > self.d  # [N, d + 1]
        rank = tf.where(ls_zero, rank + self.d + 1, rank)
        rem0 = tf.where(ls_zero, rem0 + tf.cast(self.d + 1, dtype=tf.float32), rem0)
        rank = tf.where(gt_d, rank - (self.d + 1), rank)
        rem0 = tf.where(gt_d, rem0 - tf.cast(self.d + 1, dtype=tf.float32), rem0)

        # - Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
        barycentric = tf.zeros(
            shape=[
                self.N * (self.d + 2),
            ],
            dtype=tf.float32,
        )  # [N x (d + 2), ]
        vs = tf.reshape(
            (elevated - rem0) * down_factor,
            shape=[
                -1,
            ],
        )  # [N x (d + 1), ]
        idx = tf.reshape(
            (self.d - rank) + tf.range(self.N)[..., tf.newaxis] * (self.d + 2),
            shape=[
                -1,
            ],
        )  # [N x (d + 1), ]
        idx1 = tf.reshape(
            (self.d - rank + 1) + tf.range(self.N)[..., tf.newaxis] * (self.d + 2),
            shape=[
                -1,
            ],
        )  # [N x (d + 1), ]
        barycentric = tf.tensor_scatter_nd_add(
            tensor=barycentric, indices=idx[..., tf.newaxis], updates=vs
        )  # [N x (d + 2), ]
        barycentric = tf.tensor_scatter_nd_sub(
            tensor=barycentric, indices=idx1[..., tf.newaxis], updates=vs
        )  # [N x (d + 2), ]
        barycentric = tf.reshape(
            barycentric, shape=[self.N, (self.d + 2)]
        )  # [N, d + 2]
        idx0 = tf.stack(
            [
                tf.range(self.N),
                tf.zeros(
                    [
                        self.N,
                    ],
                    dtype=tf.int32,
                ),
            ],
            axis=-1,
        )  # [N, 2]
        barycentric = tf.tensor_scatter_nd_add(
            tensor=barycentric,
            indices=idx0,
            updates=(1.0 + barycentric[..., self.d + 1]),
        )  # [N, d + 2]

        # - Compute all vertices and their offset
        canonicalT = tf.transpose(self.canonical, perm=[1, 0])  # [d + 1, d + 1]
        canonical_ext = tf.gather(params=canonicalT, indices=rank)  # [N, d + 1, d + 1]
        canonical_ext = tf.transpose(canonical_ext, perm=[0, 2, 1])  # [N, d + 1, d + 1]

        # - Get keys
        keys = (
            tf.cast(rem0[..., tf.newaxis, : self.d], dtype=tf.int32)
            + canonical_ext[..., : self.d]
        )  # [N, d + 1, d]
        keys = tf.reshape(keys, shape=[-1, self.d])  # flatten, [N x (d + 1), d]
        maxs_key, mins_key = tf.reduce_max(keys, axis=0), tf.reduce_min(
            keys, axis=0
        )  # [d, ]

        # - Get 1D coordinates
        ranges_key = maxs_key - mins_key + 1  # [d, ], max - min + 1 contains all data
        dims_key = tf.math.cumprod(
            ranges_key, exclusive=True, reverse=True
        )  # [d, ], row-major
        coords_1d = tf.reduce_sum(
            (keys - mins_key[tf.newaxis, ...]) * dims_key[tf.newaxis, ...], axis=1
        )  # [N * (d + 1), ]

        coords_1d_uniq, offsets = tf.unique(coords_1d)

        M = tf.shape(coords_1d_uniq)[0]  # represents the number of `coords_1d_uniq`.

        # - Find the neighbors of each lattice point
        # - Get the number of vertices in the lattice
        # - Create the neighborhood structure
        # - For each of d+1 axes,
        n1s = tf.tile(coords_1d_uniq[:, tf.newaxis], [1, self.d + 1]) - tf.reduce_sum(
            dims_key, axis=0
        )  # [M, d + 1]
        n2s = tf.tile(coords_1d_uniq[:, tf.newaxis], [1, self.d + 1]) + tf.reduce_sum(
            dims_key, axis=0
        )  # [M, d + 1]
        n1s = n1s + tf.reduce_sum(
            (
                self.d_mat[tf.newaxis, ..., : self.d]
                + self.diagone[tf.newaxis, ..., : self.d]
            )
            * dims_key[tf.newaxis, tf.newaxis, ...],
            axis=-1,
        )  # [M, d + 1]
        n2s = n2s - tf.reduce_sum(
            (
                self.d_mat[tf.newaxis, ..., : self.d]
                + self.diagone[tf.newaxis, ..., : self.d]
            )
            * dims_key[tf.newaxis, tf.newaxis, ...],
            axis=-1,
        )  # [M, d + 1]
        ns = tf.stack([n1s, n2s], axis=0)  # [2, M, d + 1]

        # - Shift all values by 1 such that -1 -> 0 (used for blurring)
        os = (
            tf.reshape(
                offsets,
                shape=[
                    -1,
                ],
            )
            + 1
        )  # [N x (d + 1), ]
        ws = tf.reshape(
            barycentric[..., : self.d + 1],
            shape=[
                -1,
            ],
        )  # [N x (d + 1), ]

        # - Returns
        return os, ws, coords_1d_uniq, M, ns

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None], dtype=tf.float32
            ),  # of inp, flatten, [N, value_size], float32
            tf.TensorSpec(
                shape=[
                    None,
                ],
                dtype=tf.int32,
            ),  # of os, [N x (d + 1), ], int32
            tf.TensorSpec(
                shape=[
                    None,
                ],
                dtype=tf.float32,
            ),  # of ws, [N x (d + 1), ], float32
            tf.TensorSpec(
                shape=[2, None, None], dtype=tf.int32
            ),  # of blur_neighbors, [2, M, (d + 1)], int32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of M, [], int32
            tf.TensorSpec(shape=[], dtype=tf.bool),  # of reverse, [], bool
        ]
    )
    def compute(
        self,
        inp: tf.Tensor,
        os: tf.Tensor,
        ws: tf.Tensor,
        blur_neighbors: tf.Tensor,
        M: tf.int32,
        reverse: bool = False,
    ) -> tf.Tensor:
        """
        Compute sequentially.

        Args:
            inp: entity to be filtered, [size (a.k.a. N), value_size], float32, channel-last.
            d: dimension of features, [], int32.
            alpha: magic number, [], float32.
            os: offset, [N x (d + 1), ], int32.
            ws: barycentric weight, [N x (d + 1), ], float32.
            blur_neighbors: blur neighbors, [2, M, (d + 1)], int32
            M: the number of coord_1d_uniq, [], int32
            reverse: indicating the blur order.

        Returns:
            out: [size, value_size]
        """

        value_size = tf.shape(inp)[1]

        # **************************
        # * 2022-05-26: Numpifying *
        # **************************
        # - Shift all values by 1 such that -1 -> 0 (used for blurring)

        # ->> Splat
        inpT = tf.transpose(
            inp, perm=[1, 0]
        )  # transpose to channel-first. [value_size, N]

        def splat_channelwise(ch):
            ch_ext = tf.tile(ch[..., tf.newaxis], [1, self.d + 1])  # [N, (d + 1)]
            ch_flat = tf.reshape(
                ch_ext,
                shape=[
                    -1,
                ],
            )  # [N x (d + 1), ]
            val_ch = tf.math.bincount(
                os,
                weights=ch_flat * ws,
                minlength=M + 2,
                maxlength=M + 2,
                dtype=tf.float32,
            )
            return val_ch

        valuesT = tf.vectorized_map(splat_channelwise, inpT)  # [value_size, M + 2]
        values = tf.transpose(valuesT, perm=[1, 0])  # [M + 2, value_size]

        # ->> Blur
        j_range = tf.range(self.d, -1, -1) if reverse else tf.range(self.d + 1)
        idx_nv = tf.range(1, M + 1)  # [M, ]

        for j in j_range:
            n1s = blur_neighbors[0, ..., j]  # [M, ]
            n2s = blur_neighbors[1, ..., j]  # [M, ]
            n1_vals = tf.gather(values, n1s)  # [M, value_size]
            n2_vals = tf.gather(values, n2s)  # [M, value_size]

            values = tf.tensor_scatter_nd_add(
                tensor=values,
                indices=idx_nv[..., tf.newaxis],
                updates=0.5 * (n1_vals + n2_vals),
            )

        # ->> Slice
        out = ws[..., tf.newaxis] * tf.gather(values, os) * self.alpha
        out = tf.reshape(out, shape=[-1, self.d + 1, value_size])
        out = tf.reduce_sum(out, axis=1)

        return out

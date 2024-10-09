"""
Baseline class of Refined UNet v4. 
"""
import tensorflow as tf

from .crf_computation import CRFComputation
from .rfn_unet_v4_config import RefinedUNetV4Config
from .linear2 import Linear2


class RefinedUNetV4(tf.Module):
    def __init__(
        self,
        config: RefinedUNetV4Config = None,
        ugenerator_path: str = None,
        name: str = None,
    ):
        super().__init__(name)

        self.config = config
        self.crf_computation = CRFComputation()
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
            tf.ones(shape=[self.config.height, self.config.width, 3], dtype=tf.float32)
        )
        self.crf_computation.mean_field_approximation(
            tf.ones(
                shape=[self.config.height, self.config.width, self.config.num_classes],
                dtype=tf.float32,
            ),
            tf.ones(shape=[self.config.height, self.config.width, 3], dtype=tf.float32),
            theta_alpha=self.config.theta_alpha,
            theta_beta=self.config.theta_beta,
            theta_gamma=self.config.theta_gamma,
            bilateral_compat=self.config.bilateral_compat,
            spatial_compat=self.config.spatial_compat,
            compatibility=self.config.compatibility,
            num_iterations=self.config.num_iterations,
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

        rfn = self.crf_computation.mean_field_approximation(
            unary,
            ref,
            theta_alpha=self.config.theta_alpha,
            theta_beta=self.config.theta_beta,
            theta_gamma=self.config.theta_gamma,
            bilateral_compat=self.config.bilateral_compat,
            spatial_compat=self.config.spatial_compat,
            compatibility=self.config.compatibility,
            num_iterations=self.config.num_iterations,
        )

        return rfn

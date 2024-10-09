import tensorflow as tf
import tensorflow_probability as tfp


class Linear2(tf.Module):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)]
    )
    def __call__(self, image):
        truncated_percentile = tf.constant(2, dtype=tf.float32)
        minout = tf.constant(0, dtype=tf.float32)
        maxout = tf.constant(1, dtype=tf.float32)

        def process_channelwise(ch):
            truncated_down = tfp.stats.percentile(ch, truncated_percentile)
            truncated_up = tfp.stats.percentile(ch, 100 - truncated_percentile)

            new_ch = ((maxout - minout) / (truncated_up - truncated_down)) * ch
            new_ch = tf.where(new_ch < minout, minout, new_ch)
            new_ch = tf.where(new_ch > maxout, maxout, new_ch)

            return new_ch

        imageT = tf.transpose(image, perm=[2, 0, 1])  # to channel-first
        new_imageT = tf.map_fn(fn=process_channelwise, elems=imageT)
        new_image = tf.transpose(new_imageT, perm=[1, 2, 0])
        new_image = tf.where(image > 0, new_image, 0)

        return new_image

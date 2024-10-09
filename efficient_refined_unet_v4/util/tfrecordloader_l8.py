import tensorflow as tf

def load_testset(filenames: list, batch_size: int) -> tf.data.TFRecordDataset:
    """ Load a tensorflow TFDataset file as a test set
    """
    test_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'x_train': tf.io.FixedLenFeature([], tf.string),
        'y_train': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(
            example_proto, feature_description)

        x = tf.io.decode_raw(example['x_train'], tf.int32)
        y = tf.io.decode_raw(example['y_train'], tf.uint8)

        x = tf.reshape(x, [512, 512, 7])
        y = tf.reshape(y, [512, 512])

        x = tf.cast(x, tf.float32)
        x = (x - tf.reduce_min(x)) / \
            (tf.reduce_max(x) - tf.reduce_min(x) + 1e-10)
        y = tf.cast(y, tf.int32)

        example['x_train'] = x
        example['y_train'] = y

        return example

    test_dataset = test_dataset.map(_parse_function).batch(
        batch_size, drop_remainder=True)

    return test_dataset
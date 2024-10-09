import numpy as np

def reconstruct_full(tiles: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    """ Reconstruct a full image from a tile list
    """
    def reconstruct(tiles, num_height, num_width, crop_height, crop_width):
        """ Reconstruct from a tile list. A reverse function of `extract_pairwise_patches`
        """

        assert num_height * num_width == tiles.shape[0], 'Dim is wrong. {} X {} != {}'.format(
            num_height, num_width, tiles.shape[0])

        full = np.ndarray(shape=(
            num_height * crop_height, num_width * crop_width), dtype=tiles.dtype)

        for i in range(num_height):
            for j in range(num_width):
                full[(crop_height * i):(crop_height * (i + 1)), (crop_width * j)
                            :(crop_width * (j + 1))] = tiles[i * num_width + j]

        return full

    if tiles.shape[0] == 240:
        num_height = 15
        num_width = 16
    elif tiles.shape[0] == 256:
        num_height = 16
        num_width = 16
    else:
        print('Shape of tiles error!')

    full = reconstruct(
        tiles, num_height=num_height, num_width=num_width, crop_height=crop_height, crop_width=crop_width)

    # full.shape is [num_height x 512, num_width x 512]
    return full
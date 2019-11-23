from skimage.filters import rank_order
import numpy as np

def reconstruction(seed, mask, method='dilation', selem=None, offset=None):
    assert tuple(seed.shape) == tuple(mask.shape)
    if method == 'dilation' and np.any(seed > mask):
        raise ValueError("Intensity of seed image must be less than that "
                         "of the mask image for reconstruction by dilation.")
    elif method == 'erosion' and np.any(seed < mask):
        raise ValueError("Intensity of seed image must be greater than that "
                         "of the mask image for reconstruction by erosion.")
    try:
        from ._greyreconstruct import reconstruction_loop
    except ImportError:
        raise ImportError("_greyreconstruct extension not available.")

    if selem is None:
        selem = np.ones([3] * seed.ndim, dtype=bool)
    else:
        selem = selem.astype(bool)

    if offset is None:
        if not all([d % 2 == 1 for d in selem.shape]):
            raise ValueError("Footprint dimensions must all be odd")
        offset = np.array([d // 2 for d in selem.shape])
    else:
        if offset.ndim != selem.ndim:
            raise ValueError("Offset and selem ndims must be equal.")
        if not all([(0 <= o < d) for o, d in zip(offset, selem.shape)]):
            raise ValueError("Offset must be included inside selem")

    # Cross out the center of the selem
    selem[tuple(slice(d, d + 1) for d in offset)] = False

    # Make padding for edges of reconstructed image so we can ignore boundaries
    dims = np.zeros(seed.ndim + 1, dtype=int)
    dims[1:] = np.array(seed.shape) + (np.array(selem.shape) - 1)
    dims[0] = 2
    inside_slices = tuple(slice(o, o + s) for o, s in zip(offset, seed.shape))
    # Set padded region to minimum image intensity and mask along first axis so
    # we can interleave image and mask pixels when sorting.
    if method == 'dilation':
        pad_value = np.min(seed)
    elif method == 'erosion':
        pad_value = np.max(seed)
    else:
        raise ValueError("Reconstruction method can be one of 'erosion' "
                         "or 'dilation'. Got '%s'." % method)
    images = np.full(dims, pad_value, dtype='float64')
    images[(0, *inside_slices)] = seed
    images[(1, *inside_slices)] = mask

    # Create a list of strides across the array to get the neighbors within
    # a flattened array
    value_stride = np.array(images.strides[1:]) // images.dtype.itemsize
    image_stride = images.strides[0] // images.dtype.itemsize
    selem_mgrid = np.mgrid[[slice(-o, d - o)
                            for d, o in zip(selem.shape, offset)]]
    selem_offsets = selem_mgrid[:, selem].transpose()
    nb_strides = np.array([np.sum(value_stride * selem_offset)
                           for selem_offset in selem_offsets], np.int32)

    images = images.flatten()

    # Erosion goes smallest to largest; dilation goes largest to smallest.
    index_sorted = np.argsort(images).astype(np.int32)
    if method == 'dilation':
        index_sorted = index_sorted[::-1]

    # Make a linked list of pixels sorted by value. -1 is the list terminator.
    prev = np.full(len(images), -1, np.int32)
    next = np.full(len(images), -1, np.int32)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next[index_sorted[:-1]] = index_sorted[1:]

    # Cython inner-loop compares the rank of pixel values.
    if method == 'dilation':
        value_rank, value_map = rank_order(images)
    elif method == 'erosion':
        value_rank, value_map = rank_order(-images)
        value_map = -value_map

    start = index_sorted[0]
    reconstruction_loop(value_rank, prev, next, nb_strides, start,
                        image_stride)

    # Reshape reconstructed image to original image shape and remove padding.
    rec_img = value_map[value_rank[:image_stride]]
    rec_img.shape = np.array(seed.shape) + (np.array(selem.shape) - 1)
    return rec_img[inside_slices]

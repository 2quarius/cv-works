from skimage import img_as_float
from scipy.ndimage import binary_erosion,generate_binary_structure,_ni_support,_ni_docstrings,_nd_image
from skimage._shared.utils import check_nD
import numpy as np
import cv2
import matplotlib.pyplot as plt

EROSION_SELEM = generate_binary_structure(2, 2)

HSOBEL_WEIGHTS = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]) / 4.0
VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

def sobel(image, mask=None):
    """Find the edge magnitude using the Sobel transform.
    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.
    Returns
    -------
    output : 2-D array
        The Sobel edge map.
    See also
    --------
    scharr, prewitt, roberts, feature.canny
    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.
    The 3x3 convolution kernel used in the horizontal and vertical Sobels is
    an approximation of the gradient of the image (with some slight blurring
    since 9 pixels are used to compute the gradient at a given pixel). As an
    approximation of the gradient, the Sobel operator is not completely
    rotation-invariant. The Scharr operator should be used for a better
    rotation invariance.
    Note that ``scipy.ndimage.sobel`` returns a directional Sobel which
    has to be further processed to perform edge detection.
    Examples
    # --------
    # >>> from skimage import data
    # >>> camera = data.camera()
    # >>> from skimage import filters
    # >>> edges = filters.sobel(camera)
    """
    check_nD(image, 2)
    out = np.sqrt(sobel_h(image, mask) ** 2 + sobel_v(image, mask) ** 2)
    out /= np.sqrt(2)
    return out

def sobel_h(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.
    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.
    Returns
    -------
    output : 2-D array
        The Sobel edge map.
    Notes
    -----
    We use the following kernel::
      1   2   1
      0   0   0
     -1  -2  -1
    """
    check_nD(image, 2)
    image = img_as_float(image)
    result = convolve(image, HSOBEL_WEIGHTS)
    return _mask_filter_result(result, mask)


def sobel_v(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.
    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.
    Returns
    -------
    output : 2-D array
        The Sobel edge map.
    Notes
    -----
    We use the following kernel::
      1   0  -1
      2   0  -2
      1   0  -1
    """
    check_nD(image, 2)
    image = img_as_float(image)
    result = convolve(image, VSOBEL_WEIGHTS)
    return _mask_filter_result(result, mask)

def _mask_filter_result(result, mask):
    """Return result after masking.
    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        mask = binary_erosion(mask, EROSION_SELEM, border_value=0)
        return result * mask

def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    weights = np.asarray(weights, dtype=np.float64)
    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError('filter weights array has incorrect shape.')
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not weights.shape[ii] & 1:
                origins[ii] -= 1
    for origin, lenw in zip(origins, wshape):
        if _invalid_origin(origin, lenw):
            raise ValueError('Invalid origin; origin must satisfy '
                             '-(weights.shape[k] // 2) <= origin[k] <= '
                             '(weights.shape[k]-1) // 2')

    if not weights.flags.contiguous:
        weights = weights.copy()
    output = _ni_support._get_output(output, input)
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate(input, weights, output, mode, cval, origins)
    return output
def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)

@_ni_docstrings.docfiller
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, True)
if __name__=="__main__":
    img = cv2.imread('/Users/sixplus/Desktop/test.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = sobel(gray)
    plt.figure()
    plt.imshow(edges)
    print(0)

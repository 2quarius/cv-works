# import the necessary packages
import numpy as np
import cv2

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")


def convolve(input, kernel):
    r, c = input.shape
    h, w = kernel.shape
    addw = int((w - 1) / 2)
    addh = int((h - 1) / 2)
    img = np.zeros([r + w - 1, c + h - 1])
    img[addw:addw + r, addh:addh + c] = input[:, :]
    output = np.zeros_like(a=img)

    for i in range(addw, addw + r):
        for j in range(addh, addh + c):
            output[i][j] = int(np.sum(img[i - addw:i + addw + 1, j - addw:j + addw + 1] * kernel))
    return output[addw:addw + r, addh:addh + c]


def roberts_convolve(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    roberts = [[-1, -1], [1, 1]]
    r, c = img.shape
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = roberts * imgChild
                img[x, y] = abs(list_robert.sum())
    return img


def prewitt_convolve(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    gy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])
    prewittx = convolve(img, gx)
    prewitty = convolve(img, gy)
    result = cv2.addWeighted(cv2.convertScaleAbs(prewittx), 0.5, cv2.convertScaleAbs(prewitty), 0.5, 0)
    return result


def sobel_convolve(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    r, c = img.shape
    result = np.zeros((r, c))
    x = np.zeros(img.shape)
    y = np.zeros(img.shape)
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            x[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * gx))
            y[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * gy))
            result[i + 1, j + 1] = (x[i + 1, j + 1] * x[i + 1, j + 1] + y[i + 1, j + 1] * y[i + 1, j + 1]) ** 0.5
    return np.uint8(result)

# if __name__ == "__main__":
#     image = cv2.imread('/Users/sixplus/Desktop/test.png')
#     cv2.imshow("alpha", image)
#     # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     edges = sobel_convolve(image)
#     cv2.imshow("edges", edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

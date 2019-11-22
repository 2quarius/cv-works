import cv2 as cv
import math
import numpy as np


class GaussBlur:
    def __init__(self, img, ksize, sigma):
        self.image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        self.size = ksize
        self.sigma = sigma

    def gauss(self):
        [h, w] = self.image.shape
        tmp = np.zeros((h, w), np.uint8)
        kernel = self.gausskernel()
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sum = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        sum += self.image[i + k, j + l] * kernel[k + 1, l + 1]
                tmp[i, j] = sum
        return tmp

    def gausskernel(self):
        gausskernel = np.zeros((self.size, self.size), np.float32)
        for i in range(self.size):
            for j in range(self.size):
                norm = math.pow(i - 1, 2) + pow(j - 1, 2)
                gausskernel[i, j] = math.exp(-norm / (2 * math.pow(self.sigma, 2)))
        sum = np.sum(gausskernel)
        kernel = gausskernel / sum
        return kernel


class MeanBlur:
    def __init__(self, image, fsize):
        self.image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.size = fsize

    def mean(self):
        '''
        均值滤波器
        :param input_image: 输入图像
        :param filter_size: 滤波器大小
        :return: 输出图像

        注：此实现滤波器大小必须为奇数且 >= 3
        '''
        input_image_cp = np.copy(self.image)  # 输入图像的副本

        filter_template = np.ones((self.size, self.size))  # 空间滤波器模板

        pad_num = int((self.size - 1) / 2)  # 输入图像需要填充的尺寸

        input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像

        m, n = input_image_cp.shape  # 获取填充后的输入图像的大小

        output_image = np.copy(input_image_cp)  # 输出图像

        # 空间滤波
        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                output_image[i, j] = np.sum(
                    filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) / (
                                                 self.size ** 2)

        output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

        return output_image


class MedianBlur:
    def __init__(self, image, ksize, padding=None):
        self.image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.size = ksize
        self.padding = padding

    def median(self):
        [h, w] = self.image.shape
        if not self.padding:
            edge = int((self.size - 1) / 2)
            if h - 1 - edge <= edge or w - 1 - edge <= edge:
                print("The parameter ksize is too large")
                return None
            new_arr = np.zeros((h, w), dtype="uint8")
            for i in range(edge, h - edge):
                for j in range(edge, w - edge):
                    new_arr[i, j] = np.median(self.image[i - edge:i + edge + 1, j - edge:j + edge + 1])
        return new_arr

    def colored(self):
        B, G, R = cv.split(self.image)
        # 对 蓝色通道 进行中值滤波
        H = np.zeros(256, dtype=int)  # 直方图
        for row in range(1, len(B) - 1):
            # 到达一个新的行 初始化
            H = np.zeros(256, dtype=int)  # 直方图
            # 求中值
            med = np.uint8(np.median(B[row - 1: row + 2, 0:3]))
            n = 0
            for i in range(-1, 2):
                for j in range(0, 3):
                    H[B[row + i][j]] = H[B[row + i][j]] + 1
                    if B[row + i][j] <= med:
                        n = n + 1
            for col in range(1, len(B[row]) - 1):
                if col == 1:
                    None
                # 移到下一列
                else:
                    # 更新直方图 并计算 n 的值
                    for i in range(-1, 2):
                        # 对左列元素 值减一
                        H[B[row + i][col - 2]] = H[B[row + i][col - 2]] - 1
                        if B[row + i][col - 2] <= med:
                            n = n - 1
                        # 对右列元素 值加一
                        H[B[row + i][col + 1]] = H[B[row + i][col + 1]] + 1
                        if B[row + i][col + 1] <= med:
                            n = n + 1
                    # 重新计算中值
                    if n > 5:
                        while n > 5:
                            if med == 0:
                                break
                            n = n - H[med]
                            med = med - 1
                    elif n < 5:
                        while n < 5:
                            med = med + 1
                            n = n + H[med]
                sum = 0
                for k in range(med + 1):
                    sum = sum + H[k]
                # 更新中值后的直方图
                H[B[row][col]] = H[B[row][col]] - 1
                if med < B[row][col]:
                    n = n + 1
                B[row][col] = med
                H[med] = H[med] + 1
        # 对 绿色通道 进行中值滤波
        H = np.zeros(256, dtype=int)  # 直方图
        for row in range(1, len(G) - 1):
            # 到达一个新的行 初始化
            H = np.zeros(256, dtype=int)  # 直方图
            # 求中值
            med = np.uint8(np.median(G[row - 1: row + 2, 0:3]))
            if med == -128:
                print(G[row - 1: row + 2, 0:3])
            n = 0
            for i in range(-1, 2):
                for j in range(0, 3):
                    H[G[row + i][j]] = H[G[row + i][j]] + 1
                    if G[row + i][j] <= med:
                        n = n + 1
            for col in range(1, len(G[row]) - 1):
                if col == 1:
                    None
                # 移到下一列
                else:
                    # 更新直方图 并计算 n 的值
                    for i in range(-1, 2):
                        # 对左列元素 值减一
                        H[G[row + i][col - 2]] = H[G[row + i][col - 2]] - 1
                        if G[row + i][col - 2] <= med:
                            n = n - 1
                        # 对右列元素 值加一
                        H[G[row + i][col + 1]] = H[G[row + i][col + 1]] + 1
                        if G[row + i][col + 1] <= med:
                            n = n + 1
                    # 重新计算中值
                    if n > 5:
                        while n > 5:
                            if med == 0:
                                break
                            n = n - H[med]
                            med = med - 1
                    elif n < 5:
                        while n < 5:
                            med = med + 1
                            n = n + H[med]
                # 更新中值后的直方图
                H[G[row][col]] = H[G[row][col]] - 1
                if med < G[row][col]:
                    n = n + 1
                G[row][col] = med
                H[med] = H[med] + 1
        # 对 红色通道 进行中值滤波
        H = np.zeros(256, dtype=int)  # 直方图
        for row in range(1, len(R) - 1):
            # 到达一个新的行 初始化
            H = np.zeros(256, dtype=int)  # 直方图
            # 求中值
            med = np.uint8(np.median(R[row - 1: row + 2, 0:3]))
            if med == -128:
                print(R[row - 1: row + 2, 0:3])
            n = 0
            for i in range(-1, 2):
                for j in range(0, 3):
                    H[R[row + i][j]] = H[R[row + i][j]] + 1
                    if R[row + i][j] <= med:
                        n = n + 1
            for col in range(1, len(R[row]) - 1):
                if col == 1:
                    None
                # 移到下一列
                else:
                    # 更新直方图 并计算 n 的值
                    for i in range(-1, 2):
                        # 对左列元素 值减一
                        H[R[row + i][col - 2]] = H[R[row + i][col - 2]] - 1
                        if R[row + i][col - 2] <= med:
                            n = n - 1
                        # 对右列元素 值加一
                        H[R[row + i][col + 1]] = H[R[row + i][col + 1]] + 1
                        if R[row + i][col + 1] <= med:
                            n = n + 1
                    # 重新计算中值
                    if n > 5:
                        while n > 5:
                            if med == 0:
                                break
                            n = n - H[med]
                            med = med - 1
                    elif n < 5:
                        while n < 5:
                            med = med + 1
                            n = n + H[med]
                sum = 0
                # 更新中值后的直方图
                H[R[row][col]] = H[R[row][col]] - 1
                if med < R[row][col]:
                    n = n + 1
                R[row][col] = med
                H[med] = H[med] + 1

        return cv.merge([B, G, R])

# image = cv.imread("/Users/sixplus/Desktop/test.png")
# gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# gauss = MeanBlur(gray,5)
# gaussimage = gauss.mean()
# cv.imshow("image",image)
# cv.imshow("grayimage",gray)
# cv.imshow("gaussimage",gaussimage)
# cv.waitKey(0)
# cv.destroyAllWindows()

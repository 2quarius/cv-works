import numpy as np
from PIL import Image
class ImageMorphological():
    def __init__(self, image, threshold = 120):
        '''初始化

        参数:
            image_path: 待处理的图片路径
            threshold: 阈值T
        '''
        # 读入图片
        self.image = np.asarray(image)
        # 灰度阈值变换
        self.wb = self.gray2wb(threshold=threshold)

        self.gray = np.asanyarray(image.convert('L'))
        self.bin = np.asanyarray(image.convert('1'))

    def gray2wb(self, threshold = 120):
        '''灰度阈值

        参数：
            threshold: 阈值

        返回：
            wb：阈值变换后的图像矩阵
        '''
        wb = (self.image >= threshold) * 255
        wb = wb.astype(np.uint8)
        return wb

    def imsave(self, imarray, path = './pics/wb.png'):
        '''把图像矩阵保存为图片。

        参数：
            imarray: 图像矩阵
            path: 保存路径
        '''
        tmp = Image.fromarray(np.uint8(imarray))
        tmp.save(path, 'png')
    def pad(self, img, row = 1, col = 1, mode = 'constant'):
        '''Padding an image.

        Args:
            img: 图片矩阵
            row: 行填充数
            clo: 列填充数

        Returns:
            image_pad: 填充之后的图像矩阵
        '''
        return np.pad(img, ((row, row), (col, col)), mode)

    def is_cover(self, kernel, img_slice):
        '''判断结构元素是否被包含在图像目标区域

        Args:
            kernel: 结构元素（一个numpy数组）
            img_slice: 图像矩阵的一个切片

        Returns:
            true or false
        '''
        for i, row in enumerate(kernel):
            for j, value in enumerate(row):
                if value == 255 and img_slice[i, j] == 255:
                    continue
                if value == 255 and img_slice[i, j] == 0:
                    return False
        return True

    def erosion_bin(self, img, kernel, is_save = True, save_path = './pics/erosion.png'):
        '''二值腐蚀

        Args:
            img: 图像矩阵
            kernel: 结构元素（一个numpy数组）
            is_save: Ttue to save, False not save
            save_path: 保存路径
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2)
        tmp = np.zeros(img.shape)
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                covered = self.is_cover(kernel, img_pad[r:r+w, c:c+h])
                if covered:
                    tmp[r, c] = 255
                else:
                    tmp[r, c] = 0
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    def is_interset(self, kernel, img_slice):
        '''判断结构元素和图像目标区域是否有重叠。

        Args:
            kernel: 结构元素（一个numpy数组）
            img_slice: 图像切片

        Returns:
            true or false
        '''
        for i, row in enumerate(kernel):
            for j, value in enumerate(row):
                if value == 255 and img_slice[i, j] == 255:
                    return True
                if value == 255 and img_slice[i, j] == 0:
                    continue
        return False

    def dilation_bin(self, img, kernel, is_save = True, save_path = './pics/dilation.png'):
        '''二值膨胀

        Args:
            img: 图像矩阵
            kernel: 结构元素（一个numpy数组）
            is_save: Ttue to save, False not save
            save_path: 保存路径
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2)
        tmp = np.zeros(img.shape)
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                intersected = self.is_interset(kernel, img_pad[r:r+w, c:c+h])
                if intersected:
                    tmp[r, c] = 255
                else:
                    tmp[r, c] = 0

        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    def imopen(self, img, se_erosion, se_dilation, save_path = './pics/open.png', mode = "bin"):
        '''First erosion and then dilation.

        Args:
            img: 图像矩阵
            se_erosion: 腐蚀结构元素
            se_dilation: 膨胀结构元素
            save_path: 保存路径
            mode: "bin" or "gray"
        '''
        tmp = np.zeros(img.shape)
        if mode == 'bin':
            tmp = self.erosion_bin(img, se_erosion, False)
            tmp = self.dilation_bin(tmp, se_dilation, False)
        elif mode == 'gray':
            tmp = self.erosion_gray(img, se_erosion, False)
            tmp = self.dilation_gray(tmp, se_dilation, False)
        self.imsave(tmp, save_path)

    def imclose(self, img, se_erosion, se_dilation, save_path = './pics/close.png', mode = "bin"):
        '''First dialtion and then erosion.

        Args:
            img: 图像矩阵
            se_erosion: 腐蚀结构元素
            se_dilation: 膨胀结构元素
            save_path: 保存路径
            mode: "bin" or "gray"
        '''
        tmp = np.zeros(img.shape)
        if mode == 'bin':
            tmp = self.dilation_bin(img, se_dilation, False)
            tmp = self.erosion_bin(tmp, se_erosion, False)
        elif mode == 'gray':
            tmp = self.dilation_gray(img, se_dilation, False)
            tmp = self.erosion_gray(tmp, se_erosion, False)
        self.imsave(tmp, save_path)
    def local_min(self, mask, img_slice):
        '''得出图像切片的局部极小值。

        Args:
            mask: 结构元素
            img_slice: 图像切片
        '''
        index = np.sum(mask).astype(np.int64)
        tmp = img_slice * mask
        return np.sort(tmp.reshape((1,-1)))[0][-index]

    def erosion_gray(self, img, kernel, is_save = True, save_path = './pics/erosion_gray.png'):
        '''Erosion Fucntion.

        Args:
            img: 图像矩阵
            kernel: 结构元素
            is_save: True to save, False not save
            save_path: 保存路径
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2, mode='edge')
        tmp = np.zeros(img.shape)

        for r, row in enumerate(img):
            for c, value in enumerate(row):
                tmp[r, c] = self.local_min(kernel, img_pad[r:r+w, c:c+h])
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    def local_max(self, mask, img_slice):
        '''Find the local maximum value of img_slice.

        Args:
            mask: 结构元素
            img_slice: 图像切片
        '''
        tmp = img_slice * mask
        return np.max(tmp)

    def dilation_gray(self, img, kernel, is_save = True, save_path = './pics/dilation_gray.png'):
        '''Dilation. I use the center as the origin location.

        Args:
            img: 图像矩阵
            kernel: 结构元素
            is_save: True to save, False not save
            save_path: 保存路径
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2, mode='edge')
        tmp = np.zeros(img.shape)

        for r, row in enumerate(img):
            for c, value in enumerate(row):
                tmp[r, c] = self.local_max(kernel, img_pad[r:r+w, c:c+h])
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    def std_edge_detection(self, kernel, is_save = True, save_path ='./pics/basic_gradient.png'):
        dilation = self.dilation_gray(self.gray,kernel=kernel,is_save=False)
        erosion = self.erosion_gray(self.gray,kernel=kernel,is_save=False)
        tmp = dilation-erosion
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
        return tmp
    def internal_gradient(self,kernel, is_save = True, save_path = './pics/internal_gradient.png'):
        erosion = self.erosion_gray(self.gray,kernel=kernel,is_save=False)
        tmp = self.gray-erosion
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
        return tmp
    def external_gradient(self,kernel, is_save = True, save_path = './pics/external_gradient.png'):
        dilation = self.dilation_gray(self.gray,kernel=kernel,is_save=False)
        tmp = dilation-self.gray
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
        return tmp
# if __name__ == '__main__':
#     # load image
#     img = ImageMorphological(threshold=120)
#     image = img.image
#
#     # gray2wb
#     img.imsave(img.wb, './pics/wb.png')
#     #Binary
#     # Defint 4 kernels
#     kernel1 = np.array([[0, 255, 0],
#                         [255, 255, 255],
#                         [0, 255, 0]])
#     kernel2 = np.ones((3,3))*255
#     kernel3 = np.array([[0,0,255,0,0],
#                         [0,255,255,255,0],
#                         [255,255,255,255,255],
#                         [0,255,255,255,0],
#                         [0,0,255,0,0]])
#     kernel4 = np.ones((5,5))*255

    # binary kernels erosion
    # img.erosion_bin(img.bin, kernel1, save_path='./pics/erosion_bin_k1.png')
    # img.erosion_bin(img.bin, kernel2, save_path='./pics/erosion_bin_k2.png')
    # img.erosion_bin(img.bin, kernel3, save_path='./pics/erosion_bin_k3.png')
    # img.erosion_bin(img.bin, kernel4, save_path='./pics/erosion_bin_k4.png')
    #
    # # binary kernels dilation
    # img.dilation_bin(img.bin, kernel1, save_path='./pics/dilation_bin_k1.png')
    # img.dilation_bin(img.bin, kernel2, save_path='./pics/dilation_bin_k2.png')
    # img.dilation_bin(img.bin, kernel3, save_path='./pics/dilation_bin_k3.png')
    # img.dilation_bin(img.bin, kernel4, save_path='./pics/dilation_bin_k4.png')
    #
    # img.imopen(img.bin, kernel1, kernel1, save_path='./pics/open_binary_k11.png')
    # img.imopen(img.bin, kernel2, kernel2, save_path='./pics/open_binary_k22.png')
    # img.imopen(img.bin, kernel3, kernel3, save_path='./pics/open_binary_k33.png')
    # img.imopen(img.bin, kernel4, kernel4, save_path='./pics/open_binary_k44.png')
    #
    # img.imclose(img.bin, kernel1, kernel1, save_path='./pics/close_binary_k11.png')
    # img.imclose(img.bin, kernel2, kernel2, save_path='./pics/close_binary_k22.png')
    # img.imclose(img.bin, kernel3, kernel3, save_path='./pics/close_binary_k33.png')
    # img.imclose(img.bin, kernel4, kernel4, save_path='./pics/close_binary_k44.png')

    # GRAY
    # mask1 = np.array([[0,1,0],
    #                 [1,1,1],
    #                 [0,1,0]])
    # mask2 = np.ones((3,3))
    # mask3 = np.array([[0,0,1,0,0],
    #                 [0,1,1,1,0],
    #                 [1,1,1,1,1],
    #                 [0,1,1,1,0],
    #                 [0,0,1,0,0]])
    # mask4 = np.ones((5,5))

    #gray kernels erosion
    # img.erosion_gray(img.gray, mask1, save_path='./pics/erosion_gray_k1.png')
    # img.erosion_gray(img.gray, mask2, save_path='./pics/erosion_gray_k2.png')
    # img.erosion_gray(img.gray, mask3, save_path='./pics/erosion_gray_k3.png')
    # img.erosion_gray(img.gray, mask4, save_path='./pics/erosion_gray_k4.png')
    #
    # # gray kernels dilation
    # img.dilation_gray(img.gray, mask1, save_path='./pics/dilation_gray_k1.png')
    # img.dilation_gray(img.gray, mask2, save_path='./pics/dilation_gray_k2.png')
    # img.dilation_gray(img.gray, mask3, save_path='./pics/dilation_gray_k3.png')
    # img.dilation_gray(img.gray, mask4, save_path='./pics/dilation_gray_k4.png')
    #
    # # gray kernels open
    # img.imopen(img.gray, mask1, mask1, save_path='./pics/open_gray_k11.png', mode='gray')
    # img.imopen(img.gray, mask2, mask2, save_path='./pics/open_gray_k22.png', mode='gray')
    # img.imopen(img.gray, mask3, mask3, save_path='./pics/open_gray_k33.png', mode='gray')
    # img.imopen(img.gray, mask4, mask4, save_path='./pics/open_gray_k44.png', mode='gray')
    #
    # #gray kernels close
    # img.imclose(img.gray, mask1, mask1, save_path='./pics/close_gray_k11.png', mode='gray')
    # img.imclose(img.gray, mask2, mask2, save_path='./pics/close_gray_k22.png', mode='gray')
    # img.imclose(img.gray, mask3, mask3, save_path='./pics/close_gray_k33.png', mode='gray')
    # img.imclose(img.gray, mask4, mask4, save_path='./pics/close_gray_k44.png', mode='gray')

    # gray gradient
    # img.std_edge_detection(mask1, save_path='./pics/basic_gradient_k1.png')
    # img.internal_gradient(mask1,save_path='./pics/internal_gradient_k1.png')
    # img.external_gradient(mask1,save_path='./pics/external_gradient_k1.png')
    

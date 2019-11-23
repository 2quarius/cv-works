import morphology.morphological as morphological
from PIL import Image
import cv2

def edgeDetectionStd(cimg,mask):
    pimg = Image.fromarray(cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB))
    img = morphological.ImageMorphological(pimg,threshold=120)
    return img.std_edge_detection(mask, is_save=False)

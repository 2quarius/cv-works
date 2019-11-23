import morphology.morphological as morphological
from PIL import Image
import cv2

def gradient(cimg,mask):
    pimg = Image.fromarray(cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB))
    img = morphological.ImageMorphological(pimg,threshold=120)
    tmp = img.std_edge_detection(mask, is_save=False)
    return tmp*0.5

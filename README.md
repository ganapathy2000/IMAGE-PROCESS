pip install opencv_python
pip install matplotlib

PROGRAM 1
import cv2
img=cv2.imread('butterflypic.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

PROGRAM 2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('butterflypic.jpg')
plt.imshow(img)
OUTPUT:
image


PROGRAm 3
from PIL import Image
img=Image.open("butterflypic.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

PROGRAm 4
from PIL import ImageColor
img1=ImageColor.getrgb("red")
print(img1)
OUTPUT:
(255, 0, 0)

PROGRAM 5
from PIL import ImageColor
img=Image.new('RGB',(200,400),(255,255,0))
img.show()

PROGRAM 6
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('butterflypic.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.show()

OUTPUT:
image

PROGRAM 7
from PIL import Image
image=Image.open('butterflypic.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()

OUTPUT:

Filename: butterflypic.jpg
Format: JPEG


Mode: RGB


Size: (1024, 535)
Width: 1024

Height: 535

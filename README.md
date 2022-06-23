pip install opencv_python<br>
pip install matplotlib<br>

PROGRAM 1
import cv2<br>
img=cv2.imread('butterflypic.jpg',0)<br>
cv2.imshow('image',img)
cv2.waitKey(0)<br><br>
cv2.destroyAllWindows()<br>

PROGRAM 2<br>
import matplotlib.image as mping
import matplotlib.pyplot as plt<br>
img=mping.imread('butterflypic.jpg')
plt.imshow(img)<br>
OUTPUT:
image<br>


PROGRAm 3
from PIL import Image<br>
img=Image.open("butterflypic.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

PROGRAm 4<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("red")<br>
print(img1)<br>
OUTPUT:<br>
(255, 0, 0)<br>
<br>
PROGRAM 5<br>
from PIL import ImageColor<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
<br>
PROGRAM 6
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterflypic.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.show()<br>

OUTPUT:<br>
image<br>

PROGRAM 7<br>
from PIL import Image<br>
image=Image.open('butterflypic.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

OUTPUT:<br>

Filename: butterflypic.jpg<br>
Format: JPEG<br>


Mode: RGB<br>


Size: (1024, 535)<br>
Width: 1024<br>

Height: 535<br>


1. read the file, gray scale,binary img:
import cv2<br>
img=cv2.imread('img.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
img=cv2.imread('img.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
ret, bw_img=cv2.threshold(img,127,100,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
 output:
 ![image](https://user-images.githubusercontent.com/98145098/174056152-8d6294a2-8be6-442c-bb2f-9e318d63fbc6.png)<br>
 ![image](https://user-images.githubusercontent.com/98145098/174056291-a7e7f03a-78fe-4165-afaa-be0e67ac47de.png)
Program using URL:<br><br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.industrialempathy.com/img/remote/ZiClJf-1920w.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175263174-897925ea-79fa-4f0b-9714-32782376b492.png)<br>

import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
#Reading image files<br>
imgg1= cv2.imread('imgg1.jpg') <br>
imgg2= cv2.imread('imgg2.jpg')<br>
fimg1 = imgg1 +imgg2 <br>
plt.imshow(fimg1)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175263601-c4c80528-b95e-4b57-9742-a37cc6addef6.png)<br>
<br>

fimg1 = imgg1 *imgg2 <br>
plt.imshow(fimg1)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175263713-9b509f61-b543-4a00-8b05-d8bad7fd7335.png)<br>

fimg1 = imgg1 /imgg2 <br>
plt.imshow(fimg1)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175263833-a302f1c3-fc3c-4949-afc6-6e615a256310.png)<br>

fimg1 = imgg1 -imgg2 <br>
plt.imshow(fimg1)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175263912-63b4ba1e-bb88-4ace-a47d-8279dc11c431.png)<br>


import cv2<br><br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt <br>
img=mping.imread('imggg.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175264367-da8a5a5b-c1a4-41ef-8e2f-5e48ff49bcdd.png)<br>


hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_orange =(1, 190, 200) <br>
dark_orange =(18, 255, 255)<br>
mask= cv2.inRange(hsv_img, light_orange, dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175264495-ef8ecb1c-5f8b-48a4-9abe-d9bef8122584.png)<br>


light_white =(0, 0, 200)<br>
dark_white =(145, 60, 255)<br>
mask_white=cv2.inRange(hsv_img, light_white, dark_white)<br>
result_white= cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175264617-1739f3ea-ae44-4800-9f47-2cc8f4c6e2a7.png)<br>

final_mask =mask + mask_white<br>
final_result = cv2.bitwise_and(img, img, mask=final_mask)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(final_mask, cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/175265045-3ecff3aa-924d-46d1-a29c-6f8693e6b534.png)<br>

blur =cv2.GaussianBlur (final_result, (7, 7), 8)<br>
plt.imshow(blur) <br>
plt.show()<br>



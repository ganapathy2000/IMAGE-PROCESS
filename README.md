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

import cv2<br>
img = cv2.imread("D:\download.jpg")<br>
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br> 
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/98145098/175266787-ee752c81-edf6-422a-991b-1eebec5a1ece.png)<br>

import cv2 as c<br><br>
import numpy as np<br>
from PIL import Image<br>
array = np.zeros([100, 200, 3], dtype=np.uint8)<br>
array[:,:100] =[255, 130, 0]<br>
array[:,100:] =[0, 0, 255]<br>
img= Image.fromarray(array)<br>
img.save('imgs.jpg')<br>
img.show()<br>
c.waitKey(0)<br>
output;<br>
![image](https://user-images.githubusercontent.com/98145098/175267364-b4e643a6-a2d0-4932-9ac5-6846235e4778.png)<br>

#importing Libraries<br>
import cv2<br>
import numpy as np<br>
image = cv2.imread('img2.jpg')<br>
cv2.imshow('Original Image', image)<br>
cv2.waitKey(0)<br>
# Gaussian Blur<br>
Gaussian = cv2.GaussianBlur (image, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)<br>
cv2.waitKey(0)<br>
# Median Blur<br>
median = cv2.medianBlur (image, 5) <br>
cv2.imshow('Median Blurring', median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral = cv2.bilateralFilter(image, 9, 75, 75)<br>
cv2.imshow('Bilateral Blurring', bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

Output:<br>
![image](https://user-images.githubusercontent.com/98145098/176405408-31812640-b6e7-4f00-a0f0-97bcd276f3ba.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176405517-c7a3af4f-8ab2-4fc9-aec7-d85bfc0f3344.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176405732-24ee18bc-1f0a-49d1-ab37-b43e2298b00a.png)<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('img1.jpg')<br>
image2=cv2.imread('img.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd= cv2.bitwise_and(image1, image2) <br>
bitwiseor= cv2.bitwise_or(image1, image2)<br>
bitwiseXor=cv2.bitwise_xor (image1,image2) <br>
bitwiseNot_img1= cv2.bitwise_not(image1)<br>
#bitwiseNot_img1= cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseor)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1) <br>
#plt.subplot(155)<br>
#plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/176406229-9e859bc8-2824-4f43-9c38-bcdbe36147db.png)<br>

from PIL import Image<br>
from PIL import ImageEnhance <br>
image =Image.open('img2.jpg')<br>
image.show()<br>
enh_bri =ImageEnhance.Brightness(image)<br>
brightness= 1.5<br>
image_brightened= enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col= ImageEnhance.Color(image)<br>
color= 1.5<br>
image_colored =enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con =ImageEnhance.Contrast (image) <br>
contrast = 1.5<br>
image_contrasted =enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha =ImageEnhance.Sharpness(image)<br>
sharpness =3.0<br>
image_sharped= enh_sha. enhance (sharpness)<br>
image_sharped.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/176406812-fbcca537-7bd0-4192-934f-ce4a835cb552.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176406865-d20fccaa-74c4-433f-9b8e-9c17d52f0811.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176406930-f56459ec-49be-43e0-85df-fad8359970a6.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176406974-524bdde7-9b72-451c-b637-a6396b6325be.png)<br>
![image](https://user-images.githubusercontent.com/98145098/176407015-5e838898-c9cf-4b9a-b9d1-ee16fa7ba79a.png)<br>


import cv2<br>
import numpy as np <br>
from matplotlib import pyplot as plt<br>
from PIL import Image, ImageEnhance<br>
img = cv2.imread('img2.jpg',0) <br>
ax=plt.subplots(figsize=(20, 10))<br>
kernel = np.ones((5,5), np. uint8)<br>
opening =cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion= cv2.erode(img, kernel,iterations =1)<br>
dilation = cv2.dilate(img, kernel,iterations =1)<br>
gradient = cv2.morphbr>logyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154) <br>
plt.imshow(dilation)<br>
plt.subplot(155)br>
plt.imshow(gradient)br>
cv2.waitKey(0)<br>
OUTPUTL:<br>
![image](https://user-images.githubusercontent.com/98145098/176411715-59b94b1f-284f-45fb-92dc-d913aad390c5.png)<br>

Develop a program to<br>
(i) Read the image,<br>
(ii)write (save) the grayscale image and<br>
(iii) display the original image and grayscale image<br>
import cv2<br>
OriginalImg=cv2.imread('img2.jpg')<br>
GrayImg=cv2.imread('img2.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg', GrayImg)<br>
cv2.imshow('Display Original Image', OriginalImg)<br>
cv2.imshow('Display Grayscale Image', GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
 print('The image is successfully saved.')<br>
 
 OUTPUT:<br>
 ![image](https://user-images.githubusercontent.com/98145098/178697788-b49da998-8816-44ff-9f36-a3cdb2797190.png)<br>
 ![image](https://user-images.githubusercontent.com/98145098/178697879-7a4827f8-2337-42f2-9564-91752f95afcf.png)<br>


GrayScale slicing with background:<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('img2.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150): <br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/178699907-b6d869fd-cd0f-412e-bcfc-a601e8999133.png)<br>


GrayScale slicing without background:<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('img2.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y): <br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/178707085-2835d7eb-8f29-464b-b67c-6482705ee35b.png)


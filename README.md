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

import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('img2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('img2.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145098/178962784-88f80b28-7dd8-4cfb-b945-5d8b309b8c43.png)<br>

Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>

%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings ("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('img1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179945068-88514aab-e250-4b5c-b082-e548264073f6.png)
![image](https://user-images.githubusercontent.com/98144065/179947690-79f0a173-5756-4a23-83a1-df1fb000e5d0.png)<br><br>

prg25<br>
negative =255- pic #neg = (L-1) img <br>
plt.figure(figsize= (6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179945355-bee6e5d0-710e-4c64-b48a-d7b878680f6c.png)<br><br>

prg26<br><br>
%matplotlib inline<br>
import imageio <br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('img1.jpg') <br>
gray=lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587,0.114]) <br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(), cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179948638-b4d41d22-5899-487d-a2ef-b14d60ee944c.png)<br>
<br>
prg27<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
#Gamma encoding<br>
pic=imageio.imread('img3.jpg')<br>
gamma=2.2 #Gamma < 1 ~ Dark; Gamma > 1~ Bright<br>

gamma_correction=((pic/255)**(1/gamma)) <br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/179949121-b3a06a89-a30b-41e4-b8fa-bd8ba86b633a.png)<br><br>
prg28<br><br>
Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>
from PIL import Image<br><br>
from PIL import ImageFilter <br><br>
import matplotlib.pyplot as plt<br><br>
 
my_image = Image.open('img3.jpg')<br><br>
sharp= my_image.filter(ImageFilter.SHARPEN)<br><br>

sharp.save('D:/i2.jpg')<br><br>
sharp.show() <br><br>
plt.imshow(sharp)<br><br>
plt.show()<br><br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/179951827-c47ff40b-92de-40c4-96de-0e636ada6b2d.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/179952229-0dc22517-dcf6-4fd4-9b9b-fda9e4c1e24a.png)<br><br>
prg29<br><br>
#Image flip<br><br>
import matplotlib.pyplot as plt<br><br>
#Load the image<br><br>
img = Image.open('img3.jpg')<br><br>
plt.imshow(img) <br><br>
plt.show()<br><br>
#use the flip function<br><br>
flip = img.transpose(Image.FLIP_LEFT_RIGHT)<br><br>
#save the image<br><br>
flip.save('D:/image_flip.jpg')<br><br>
plt.imshow(flip)<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179952438-e40c8837-ea1c-4903-9a43-5ea663ee8e2a.png)<br>
![image](https://user-images.githubusercontent.com/98144065/179952464-a9a98c5d-4c30-46a9-98d3-6ae2eaa5a44c.png)<br>
![image](https://user-images.githubusercontent.com/98144065/179952614-67099e11-9ec2-4e6c-ba86-601daf7a80fe.png)<br>

prg30<br>
# Importing Image class from PIL module <br>
import matplotlib.pyplot as plt # Opens a image in RGB mode <br>
im=Image.open('img3.jpg')<br>
# Size of the image in pixels (size of original image) #(This is not mandatory)<br>
width, height = im.size<br>
#Cropped image of above dimension # (It will not change original image) <br>
im1=im.crop ((280, 100, 800, 600))<br>
#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/179952541-b0492478-04de-4903-835b-693f1d7c49fc.png)<br>

**28.Generate matrix and display the image data.**<br>
import matplotlib.image as image<br>
img=image.imread('teddy.jpg')<br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180191181-cc3ac009-6deb-411f-8a0f-bb155abd9c42.png)<br>
![image](https://user-images.githubusercontent.com/98141711/180191269-fcc1e6c7-f8ed-4d57-aa95-c293c1393974.png)<br>
<br>
**29.program to find the brightness of a image from distance to center.**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0,0,0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
        plt.imshow(arr, cmap='gray')<br>
        plt.show()<br>
 <br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180199036-65a6fa89-d817-4295-b4e7-f158ec5f7311.png)<br>
<br>
**30.program to display the different color in diagonal with matrix.**<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180200493-6c5de916-8cf8-42bf-8827-2cf129b93945.png)<br>
<br>
<br>
<br>
<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 600, 600<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400,300:400]=[0,0,255]<br>
data[400:500,400:500]=[255,255,0]<br>
data[500:600,500:600]=[0,255,255]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**31.Read an image to find max,min,average and standard deviation of pixel value.**<br>
# example of pixel normalization<br>
from numpy import asarray<br>
from PIL import Image<br>
# load image<br>
image = Image.open('bird.jpg')<br>
pixels = asarray(image)<br>
# confirm pixel range is 0-255<br>
#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
# convert from integers to floats<br>
pixels = pixels.astype('float32')<br>
# normalize to the range 0-1<br>
pixels /= 255.0<br>
# confirm the normalization<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
<br>
**output:-**<br>
Min: 0.000, Max: 255.000<br>
Min: 0.000, Max: 1.000<br>
<br>
**Average**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("bird.jpg",0)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/181432990-7bbd99c2-054f-4555-9c12-cb35be2100d0.png)<br>
**Standard deviation**<br>
from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('bird.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/181433890-08698c2c-c54b-47dd-9449-9eef70b80108.png)<br>
**Max**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('bird.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
print(max_channels)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/181434482-85de3612-aa2f-45e6-a72c-590a4ad2a281.png)<br>
**Min**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('bird.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
print(min_channels)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/181435167-c3ccc1e9-189c-460b-9a57-71ecc17e420f.png)<br>
<br>
**32.Sobel edge and canny edge detection**<br>
import cv2<br>
# Read the original image<br>
img = cv2.imread('lion.jpg')<br>
# Display original image<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>
# Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
# Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>
# Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
# Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
# Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
# Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186391360-24246955-1ef1-49ec-b83f-83daf30efd01.png)<br>
![image](https://user-images.githubusercontent.com/98141711/186391763-ed80cda1-5e43-4b1d-9765-2e1bc3cdbdc4.png)<br>
![image](https://user-images.githubusercontent.com/98141711/186392555-3d6cf84d-3c50-4eb7-9163-ad6ee6f7d0cf.png)<br>
![image](https://user-images.githubusercontent.com/98141711/186392687-9363b946-acd6-4605-8263-3b97c7dffb5d.png)<br>
![image](https://user-images.githubusercontent.com/98141711/186392877-3cf6e8a1-f2c2-4b7b-a28a-78d0b6104bb0.png)<br>
<br>
<br>
**33.Image filtering.**<br>
import matplotlib.pyplot as plt<br>
%matplotlib inline<br>
from skimage import data,filters<br>
image = data.coins()<br>
# ... or any other NumPy array!<br>
edges = filters.sobel(image)<br>
plt.imshow(edges, cmap='gray')<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186399881-ed6fb19a-8843-4571-abd2-fd5a841c3195.png)<br>
**34.Basic pillow functions.**<br>
from PIL import Image,ImageChops,ImageFilter<br>
from matplotlib import pyplot as plt<br>
x=Image.open("x.png")<br>
o=Image.open("o.png")<br>
print('size of the image:',x.size,'colour mode:',x.mode)<br>
print('size of the image:',o.size,'colour mode:',o.mode)<br>
plt.subplot(121),plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122),plt.imshow(o)<br>
plt.axis('off')<br>
merged=ImageChops.multiply(x,o)<br>
add=ImageChops.add(x,o)<br>
greyscale=merged.convert('L')<br>
greyscale<br>
<br>
**Output:-**<br>
size of the image: (256, 256) colour mode: RGB<br>
size of the image: (256, 256) colour mode: RGB<br>
![image](https://user-images.githubusercontent.com/98141711/186625069-e9ed4862-7a76-496c-b67d-8ec4d9f961ef.png)<br>
<br>
image=merged<br>
print('image size:',image.size,<br>
     '\ncolor mode:',image.mode,<br>
      '\nimage width:',image.width,'\also represented by:',image.size[0],<br>
      '\nimage height:',image.height,'\also represented by:',image.size[1],)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186627311-2121de02-7914-4874-a113-a51ae86357b3.png)<br>
<br>
pixel=greyscale.load()<br>
for row in range(greyscale.size[0]):<br>
    for column in range(greyscale.size[1]):<br>
        if pixel[row,column]!=(255):<br>
         pixel[row,column]=(0)<br>
        <br>
greyscale<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186633228-5cf7ad99-0598-49f0-b0bd-0f6fa0e41d6e.png)<br>
<br>
invert=ImageChops.invert(greyscale)<br>
bg=Image.new('L',(255,256),color=(255))<br>
subt=ImageChops.subtract(bg,greyscale)<br>
rotate=subt.rotate(45)<br>
rotate<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186630051-e87fea3c-b913-49b4-b9cd-9bf6ff54b6a1.png)<br>
<br>
blur=greyscale.filter(ImageFilter.GaussianBlur(radius=1))<br>
edge=blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186630933-9dab7218-591f-4700-9859-4d29df070062.png)<br>
<br>
edge=edge.convert('RGB')<br>
bg_red=Image.new('RGB',(256,256),color=(255,0,0))<br>
filled_edge=ImageChops.darker(bg_red,edge)<br>
filled_edge<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/186631899-84efc81d-9e3f-49dc-aef6-50b9a8ecb449.png)<br>
<br>
edge.save('processed.png')<br>
<br>
**35.Image restoration: **<br>
(a) Restore a damaged image<br>
import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('img1.png')<br>
plt.imshow(img)<br>
plt.show()<br>
mask=cv2.imread('img2.png',0)<br>
plt.imshow(mask)<br>
plt.show()<br>
dst=cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)<br>
cv2.imwrite('dimage_inpainted.png',dst)<br>
plt.imshow(dst)<br>
plt.show()<br><br>
**OUTPUT:-**<br>
  ![image](https://user-images.githubusercontent.com/98141711/186654955-de13a36b-8e1a-4dbc-8a45-0ee9d2a8eb9c.png)<br>
    ![image](https://user-images.githubusercontent.com/98141711/186655138-a9ec0c1a-7851-4ef6-8b68-a03b30d23738.png)<br>
    (b) Removing Logo’s:<br>
    import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
from skimage.restoration import inpaint<br>
from skimage.transform import resize <br>
from skimage import color<br>
plt.rcParams['figure.figsize']=(10,8)<br>
def show_image(image,title='Image',cmap_type='gray'):<br>
    plt.imshow(image,cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
def plot_comparison(img_original,img_filtered,img_title_filtered):<br>
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,8),sharex=True,sharey=True)<br>
    ax1.imshow(img_original,cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered,cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
image_with_logo=plt.imread("imlogo.png")<br>
mask=np.zeros(image_with_logo.shape[:-1])<br>
mask[210:272, 360:425] = 1<br>
image_logo_removed=inpaint.inpaint_biharmonic (image_with_logo, mask,multichannel=True)<br>
plot_comparison (image_with_logo, image_logo_removed, "Image with logo removed")<br>
**Output:-**<br>
    ![image](https://user-images.githubusercontent.com/98141711/186655366-314c2393-047f-410f-9ab9-948617058533.png)<br>
   **Noise:**<br>
    (a) Adding noise<br>
    from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>
noisy_image=random_noise (fruit_image)<br>
plot_comparison (fruit_image, noisy_image, "Noisy image")<br>
   **Output:-**<br>
    ![image](https://user-images.githubusercontent.com/98141711/186655732-4092c46b-3743-45f4-ac41-5a3bba6811fc.png)<br>
   (b) Reducing Noise<br>
    from skimage.restoration import denoise_tv_chambolle<br>
noisy_image = plt.imread('noisy.jpg')<br>
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>
plot_comparison(noisy_image, denoised_image, 'Denoised Image')<br>
    **Output:-**<br>
    ![image](https://user-images.githubusercontent.com/98141711/186655898-9ff756af-6256-4b2f-aeb8-be79f2b94a7b.png)<br>
    (c) Reducing Noise while preserving edges<br>
    from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>
denoised_image=denoise_bilateral (landscape_image, multichannel=True)<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>
    **Output:-**<br>
    ![image](https://user-images.githubusercontent.com/98141711/186656102-2be89ecb-4be4-4429-8973-680908b88e3b.png)<br>
  **Segmentation :**<br>
    (a) Superpixel Segmentation<br>
    from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
face_image = plt.imread('face.jpg')<br>

segments = slic (face_image, n_segments=400)<br>

segmented_image = label2rgb(segments, face_image, kind='avg')<br>
plt.title('Original Image')<br>
plt.imshow(face_image)<br>
plt.show()<br>
plt.title('Segmented Image')<br>
plt.imshow((segmented_image* 1).astype(np.uint8))<br>
plt.show()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187902142-481d7cda-487c-423d-a1c8-4b2adc4b7e6e.png)<br>
**Contours:**<br>
  (a) Contouring shapes<br>
  def show_image_contour(image,contours):<br>
    plt.figure()<br>
    for n,contour in enumerate(contours):<br>
        plt.plot(contour[:,1],contour[:,0],linewidth=3)<br>
    plt.imshow(image,interpolation='nearest',cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
from skimage import measure,data<br>
horse_image=data.horse()<br>
contours=measure.find_contours(horse_image,level=0.8)<br>
show_image_contour(horse_image,contours)<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187877805-0545d05b-18af-4236-bbfd-22b82508d987.png)<br>
<br>
(b) Find contours of an image that is not binary<br>
from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>
image_dices=imread('diceimg.png')<br>
image_dices=color.rgb2gray(image_dices)<br>
thresh=threshold_otsu(image_dices)<br>
binary=image_dices>thresh<br>
contours=measure.find_contours(binary,level=0.8)<br>
show_image_contour(image_dices,contours)<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187880034-571a85d2-5f95-4aef-99f6-227570cb135a.png)<br>
(c) Count the dots in a dice's image<br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>
max_dots_shape = 50<br>
dots_contours = [cnt for cnt in contours if np. shape (cnt)[0] < max_dots_shape]<br>
show_image_contour (binary, contours)<br>
print('Dices dots number: {}.'.format(len(dots_contours)))<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187890077-d9b83e35-d65f-46c5-88c2-ef84778245b8.png)<br>
<br>
**36.Implement a program to perform various edge detection techniques**<br>
a) Canny Edge detection<br>
#Canny Edge detection<br>
import cv2<br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
plt.style.use('seaborn')<br>
loaded_image = cv2.imread("animated.jpeg")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br>
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>
plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image,cmap="gray")<br>
plt.title("original Image")<br>
plt.axis("off")<br>
plt.subplot(1,3,2)<br>
plt.imshow(gray_image, cmap="gray")<br>
plt.axis("off")<br>
plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>
plt.title("Canny Edge Detected Image")<br>
plt.show()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187898399-3c66fa2e-2d1b-49bd-9b73-88ce42fdbb28.png)<br>
<br>
b) Edge detection schemes - the gradient (Sobel - first order derivatives) based edge detector and the Laplacian (2nd order derivative, so it is extremely sensitive to noise) based edge detector.<br>
import cv2<br>
import numpy as np <br>
from matplotlib import pyplot as plt<br>
img0=cv2.imread('animated.jpeg',)<br>
gray= cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)<br>
img= cv2.GaussianBlur (gray, (3,3),0)<br>
laplacian= cv2.Laplacian (img,cv2.CV_64F)<br>
sobelx = cv2.Sobel (img,cv2.CV_64F,1,0,ksize=5) <br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) <br>
plt.subplot(2,2,1), plt.imshow(img, cmap = 'gray')<br>
plt.title('Original'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,2), plt.imshow(laplacian,cmap = 'gray') <br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')<br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray')<br>
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])<br>
plt.show()<br>
**OUTPUT:-**
![image](https://user-images.githubusercontent.com/98141711/187899484-5716c42d-9bd2-4a5a-8c3f-0fdf535b965d.png)<br>
c) Edge detection using Prewitt Operator<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
img = cv2.imread('animated.jpeg')<br>
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) <br>
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)<br>
kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])<br>
kernely=np.array([[-1,0,1], [-1,0,1],[-1,0,1]]) <br>
img_prewittx= cv2.filter2D(img_gaussian, -1, kernelx) <br>
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)<br>
cv2.imshow("Original Image", img)<br>
cv2.imshow("Prewitt x", img_prewittx)<br>
cv2.imshow("Prewitt y", img_prewitty)<br>
cv2.imshow("Prewitt", img_prewittx + img_prewitty)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187903451-07fca319-5992-4cd6-aad7-83ee173f7f77.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903539-69d7953e-df4d-4542-8787-228590034af0.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903619-71a5c34a-d089-481e-b424-b8f3bb60b9e0.png)<br>
![image](https://user-images.githubusercontent.com/98141711/187903690-a9d951a7-c0f1-442b-b4bd-c03873c5532d.png)<br>
<br>
d) Roberts Edge Detection- Roberts cross operator<br>
import cv2<br>
import numpy as np<br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt <br>
roberts_cross_v = np.array([[1, 0],<br>
                            [0,-1]])<br>
roberts_cross_h= np.array([[0, 1],<br>
                           [-1,0]])<br>
img= cv2.imread("animated.jpeg",0).astype('float64')<br>
img/=255.0<br>
vertical= ndimage.convolve( img, roberts_cross_v)<br>
horizontal=ndimage.convolve( img, roberts_cross_h)<br>
edged_img= np.sqrt( np.square (horizontal) + np.square(vertical))<br>
edged_img*=255<br>
cv2.imwrite("output.jpg",edged_img)<br>
cv2.imshow("OutputImage", edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllwindows()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/187904755-c8619d95-810b-49c3-b3bf-e485c8ad7d3a.png)<br>







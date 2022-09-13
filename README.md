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






1.Program to display grayscale image using read and write operation.
import cv2
img=cv2.imread('dog.jpg',0)
cv2.imshow('dog',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:

Grayscale

2.Program to display the image using matplotlib.
import matplotlib.image as mping
import matplotlib.pyplot as plt
img = mping.imread('Car.jpg')
plt.imshow(img)

OUTPUT:

MatPlotLib

3.Program to perform linear transformation rotation.
import cv2
from PIL import Image
img=Image.open("vase.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:

ImgRotate

4.Program to convert color string to RGB color value.
from PIL import ImageColor
#using getrgb for yellow
img1=ImageColor.getrgb("yellow")
print(img1)
#using getrgb for red
img2=ImageColor.getrgb("red")
print(img2)

OUTPUT:

ColorToRGBvalues

5.Program to create image using colors.
from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()

OUTPUT:

ImgUsingColors

6.Program to visualize the image using various color spaces.
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('ball.jpg')
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

OUTPUT:

ImgVisualization

7.Program to display the image attributes.
from PIL import Image
image=Image.open('baby.jpg')
print("Filename:", image.filename)
print("Format:", image.format)
print("Mode:", image.mode)
print("Size:", image.size)
print("Width:", image.width)
print("Height:", image.height)
image.close()

OUTPUT:

ImgAttributes

8.Program to convert original image to grayscale and binary.
import cv2
#read the image file
img=cv2.imread('sunflower.jpg')
cv2.imshow('RGB',img)
cv2.waitKey(0)

#Gray scale

img=cv2.imread('sunflower.jpg',0)
cv2.imshow('gray',img)
cv2.waitKey(0)

#Binary image

ret, bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary', bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:

rgb Grayscale binary

9.Program to Resize the original image.
import cv2
img=cv2.imread('pineapple.jpg')
print('Length and Width of Original image', img.shape)
cv2.imshow('original image',img)
cv2.waitKey(0)

#to show the resized image

imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Length and Width of Resized image',imgresize.shape)
cv2.waitKey(0)

OUTPUT:

OrgIMG

Length and Width of Original image (340, 453, 3)

ResizedIMG

Length and Width of Resized image (160, 150, 3)

10.Develop a program to readimage using URL.
pip install Scikit-Learn
import skimage
print(skimage.version)
start:
from skimage import io
import matplotlib.pyplot as plt
url='https://th.bing.com/th/id/OIP.qVuoApbCGfaPbNRSX8SmIwHaGA?w=213&h=180&c=7&r=0&o=5&pid=1.7.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()
OUTPUT: URL

11.Program to mask and blur the image.
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('Orangef.jpg')
plt.imshow(img)
plt.show()

OUTPUT: Msk1

hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(hsv_img, light_green, dark_green)
result=cv2.bitwise_and(img ,img, mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()

OUTPUT: Msk2

light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img, light_white, dark_white)
result_white=cv2.bitwise_and(img, img, mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()

OUTPUT: Msk3

final_mask = mask + mask_white
final_result = cv2.bitwise_and(img, img, mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()

OUTPUT: Msk4

blur = cv2.GaussianBlur(final_result, (7,7),0)
plt.imshow(blur)
plt.show()

OUTPUT: Msk5

12.program to perform arithmatic operations on images.
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

#Reading image files
img1 = cv2.imread('Car.jpg')
img2 = cv2.imread('Car1.jpg')

#Applying Numpy addition on images
fimg1 = img1 + img2
plt.imshow(fimg1)
plt.show()

OUTPUT: Arith1

#Saving the output image
cv2.imwrite('output.jpg', fimg1)
fimg2 = img1 - img2
plt.imshow(fimg2)
plt.show()

OUTPUT: Arith2

#Saving the output image
cv2.imwrite('output.jpg', fimg2)
fimg3 = img1 * img2
plt.imshow(fimg3)
plt.show()

OUTPUT: Arith3

#Saving the output image
cv2.imwrite('output.jpg', fimg3)
fimg4 = img1 / img2
plt.imshow(fimg4)
plt.show()

#Saving the output image
cv2.imwrite('output.jpg', fimg4)

OUTPUT: Arith4

13.Program to change the image to different color spaces.
import cv2
img = cv2.imread('D:\img.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image", gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image", hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT: Gray HSV LAB HLS YUV

14.Program to create an image using 2D array.
import cv2 as c
import numpy as np
from PIL import Image
array = np.zeros([100,200,3], dtype = np.uint8)
array[:,:100] = [255,130,0]
array[:,100:] = [0,0,255]
img = Image.fromarray(array)
img.save('image1.png')
img.show()
c.waitKey(0)

OUTPUT:2D

15.Program to perform bitwise operations on an image.
import cv2
import matplotlib.pyplot as plt
image1=cv2.imread('bird1.jpg')
image2=cv2.imread('bird1.jpg')
ax=plt.subplots(figsize=(15,10))
bitwiseAnd = cv2.bitwise_and(image1,image2)
bitwiseOr = cv2.bitwise_or(image1,image2)
bitwiseXor = cv2.bitwise_xor(image1,image2)
bitwiseNot_img1 = cv2.bitwise_not(image1)
bitwiseNot_img2 = cv2.bitwise_not(image2)
plt.subplot(151)
plt.imshow(bitwiseAnd)
plt.subplot(152)
plt.imshow(bitwiseOr)
plt.subplot(153)
plt.imshow(bitwiseXor)
plt.subplot(154)
plt.imshow(bitwiseNot_img1)
plt.subplot(155)
plt.imshow(bitwiseNot_img2)
cv2.waitKey(0)

OUTPUT:Bitwise

16.Program to perform blur operations.
import cv2
import numpy as np
image = cv2.imread('glass1.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)

Gaussian = cv2.GaussianBlur(image, (7, 7),0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)

median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)

bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT: Blur1 Blur2 Blur3 Blur4

17.Program to enhance an image.
from PIL import Image
from PIL import ImageEnhance
image = Image.open('bird2.jpg')
image.show()

#Brightness
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
image_brightened.show()

#Color
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored = enh_col.enhance(color)
image_colored.show()

#Contrast
enh_con = ImageEnhance.Contrast(image)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()

#Sharpen
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 1.5
image_sharped = enh_sha.enhance(sharpness)
image_sharped.show()

OUTPUT: IMGenhance IMG_enhance

18.Program to morph an image.
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
img = cv2.imread('tree.jpg',0)
ax = plt.subplots(figsize=(20,10))
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img, kernel, iterations = 1)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.subplot(151)
plt.imshow(opening)
plt.subplot(152)
plt.imshow(closing)
plt.subplot(153)
plt.imshow(erosion)
plt.subplot(154)
plt.imshow(dilation)
plt.subplot(155)
plt.imshow(gradient)
cv2.waitKey(0)

OUTPUT: Enhance2

19.Program to
(i)Read the image, convert it into grayscale image
(ii)write (save) the grayscale image and(iii)
import cv2
OriginalImg=cv2.imread('Chick.jpg')
GrayImg=cv2.imread('Chick.jpg',0)
isSaved=cv2.imwrite('D:/i.jpg', GrayImg)
cv2.imshow('Display Original Image', OriginalImg)
cv2.imshow('Display Grayscale Image',GrayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
if isSaved:
print('The image is succrssfully saved')

OUTPUT: chick_org chick_gray

20.Program to perform Graylevel slicing with background.
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('man.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
for j in range(0,y):
if(image[i][j]>50 and image[i][j]<150):
z[i][j]=255
else:
z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()

OUTPUT: graylevl_bg

21.Program to perform Graylevel slicing without background.
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('man.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
for j in range(0,y):
if(image[i][j]>50 and image[i][j]<150):
z[i][j]=255
else:
z[i][j]=0
equ=np.hstack((image,z))
plt.title('Graylevel slicing without background')
plt.imshow(equ,'gray')
plt.show()

OUTPUT: graylevl_wo_bg

22.Program to analyze the image data using Histogram.
//openCV
import cv2
import numpy as np
img = cv2.imread('man.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.hist(img.ravel(),256,[0,256])
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
plt.show()

OUTPUT: hist_openCV

//skimage
from skimage import io
import matplotlib.pyplot as plt
image = io.imread('man.jpg')
ax = plt.hist(image.ravel(), bins = 256)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
plt.show()

OUTPUT: hist_skimage

23.Program to perform basic image data analysis using intensity transformation:
a) Image negative
b) Log transformation
c) Gamma correction
%matplotlib inline
import imageio
import matplotlib.pyplot as plt
#import warnings
#import matplotlib.cbook
#warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread("violet.jpg")
plt.figure(figsize=(6,6))
plt.imshow(pic);
plt.axis('off');

OUTPUT: v_org

negative = 255 - pic
plt.figure(figsize=(6,6))
plt.imshow(negative);
plt.axis('off');

OUTPUT: v_neg

%matplotlib inline

import cv2
import numpy as np
import matplotlib.pyplot as plt

pic=cv2.imread('violet.jpg')
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)

max=np.max(gray)

def logtransform():
return(255/np.log(1+max))*np.log(1+gray)
plt.figure(figsize=(5,5))
plt.imshow(logtransform(), cmap=plt.get_cmap(name='gray'))
plt.axis('off');

OUTPUT: v_log

Gamma encoding
import imageio
import matplotlib.pyplot as plt
pic=imageio.imread('violet.jpg')
gamma=2.2 # Gamma < 1 = Dark; Gamma > 1 = Bright

gamma_correction = ((pic/255)**(1/gamma))
plt.figure(figsize=(5,5))
plt.imshow(gamma_correction)
plt.axis('off');

OUTPUT: v_gamma

24.Program to perform basic image manipulation:
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>

#Image Sharpen<br>
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
#Load the image
my_image=Image.open('dog1.jpg')
#Use sharpen function
sharp=my_image.filter(ImageFilter.SHARPEN)
#Save the image
sharp.save('E:/image_sharpen.jpg')
sharp.show()
plt.imshow(sharp)
plt.show()

OUTPUT: dog1

E_dog

#Image flip
import matplotlib.pyplot as plt
#Load the image
img=Image.open('dog1.jpg')
plt.imshow(img)
plt.show()
#use the flip function
flip=img.transpose(Image.FLIP_LEFT_RIGHT)
#save the image
flip.save('E:/image_flip.jpg')
plt.imshow(flip)
plt.show()

OUTPUT: E_dogg
dog_flip

Image crop
#Importing Image class from
from PIL import Image
import matplotlib.pyplot as plt
im=Image.open('dog1.jpg')
width,height=im.size
im1=im.crop((280,100,800,600))
im1.show()
plt.imshow(im1)
plt.show()

OUTPUT:
img_crop


25.Program to perform edge detection:
#Canny Edge detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

loaded_image = cv2.imread("mario.jpg")
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show()

OUTPUT:
Screenshot 2022-09-01 163313

#Laplacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = cv2.imread('mario.jpg',)

#converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

#remove noise
img= cv2.GaussianBlur(gray,(3,3),0)

#convolute with proper kernels
laplacian= cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y

plt.subplot(2,2,1), plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

OUTPUT:
Screenshot 2022-09-01 163340

#Edge detection using Prewitt operator
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mario.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)

#prewitt
kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D (img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D (img_gaussian, -1, kernely)
cv2.imshow("Original Image", img)
cv2.imshow("Prewitt x", img_prewittx)
cv2.imshow("Prewitt y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.waitKey()
cv2.destroyAllwindows()

OUTPUT:
Screenshot 2022-09-01 162418

#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

roberts_cross_v = np.array([[1, 0 ],[0,-1 ]] )
roberts_cross_h = np.array([[ 0, 1 ], [-1, 0 ]])

img= cv2.imread("mario.jpg",0).astype('float64')
img/=255.0
vertical=ndimage.convolve( img, roberts_cross_v )
horizontal=ndimage.convolve( img, roberts_cross_h)


edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
edged_img*=255
cv2.imwrite("output.jpg", edged_img)
cv2.imshow("OutputImage", edged_img)
cv2.waitKey()
cv2.destroyAllwindows()

OUTPUT:
Screenshot 2022-09-01 163123

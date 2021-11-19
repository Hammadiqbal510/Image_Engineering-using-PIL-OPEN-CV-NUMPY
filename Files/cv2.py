#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:



img2=cv2.imread("/Users/hammad/Downloads/as.jpeg")


# In[5]:


img2


# In[15]:


img2.shape


# In[8]:


cv2.imshow(" ",img2)
cv2.waitKey(1)#run if you want to show
cv2.destroyWindow()


# In[16]:


b,g,r=cv2.split(img2)


# In[17]:


b


# In[ ]:


g


# In[ ]:


r


# In[22]:


# cv2.imshow("blue",b)
plt.imshow(b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[ ]:


img3=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


# In[ ]:


img4=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# In[ ]:


cv2.imshow("",img3)
#cv2.waitKey(1)


# In[ ]:


ham=cv2.imwrite("ham.jpeg",img3)


# In[ ]:


cv2.imshow("",ham)
cv2.waitKey(1)


# In[23]:


img2=cv2.imread("/Users/hammad/Downloads/as.jpeg")


# In[24]:


img2.shape


# In[35]:


#access rows and columns
corner=img2[300:500,0:100]
plt.imshow(corner)


# In[36]:


display("corner",corner)


# In[6]:


cv2.imshow("",img2)


# In[ ]:


cv2.waitKey(1)
cv2.destroyAllWindows()


# In[8]:


cv2.imshow("corner",corner)


# In[9]:


cv2.waitKey(0)


# In[ ]:


def display(name,img2):
    cv2.imshow(name,img2)
    cv2.waitkey(1)
    cv2.destroyWindow()


# In[42]:



green=(0,255,0)
img2[0:100,0:100]=green
display("maniplute",img2)


# In[11]:


cv2.imshow("corner",green)


# In[ ]:


cv2.waitKey(1)


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('/Users/hammad/Desktop/image_03.jpeg')
img2 = cv2.imread('/Users/hammad/Desktop/image_03.jpeg')
v_img = cv2.vconcat([img1, img2])
cv2.imshow('Vertical', v_img)
cv2.waitKey(1)
cv2.destroyAllWindows()
"""h_img = cv2.hconcat([img1, img2])
cv2.imshow('Horizontal', h_img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


# In[45]:


from PIL import Image,ImageOps
import imagehash
import matplotlib.pyplot


# In[3]:


img4=Image.open('/Users/hammad/Downloads/Landscape_2.jpeg')
img5=Image.open('/Users/hammad/Downloads/Landscape_3.jpeg')


# In[4]:


img4=ImageOps.exif_transpose(img4)
img5=ImageOps.exif_transpose(img5)


# In[5]:


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


# In[7]:


get_concat_h(img4, img5).save('/Users/hammad/Desktop/hammad.jpg')


# In[ ]:


con_img=Image.open("hammad.jpg")
hash=imagehash.average_hash(con_img)
print(hash)


# In[37]:


canvas=np.zeros((300,300,3),dtype=np.uint8)


# In[45]:


def display(name,img2):
    cv2.imshow(name,img2)
    cv2.waitKey(1)
    cv2.destroyWindow(name)


# In[46]:


display("canvas",canvas)


# In[49]:



green=(0,255,0)
cv2.line(canvas,(0,0),(300,300),green)
display("maniplute",canvas)


# In[52]:


blue=(255,0,0)
cv2.rectangle(canvas,(20,20),(80,80),blue)
display("maniplute",canvas)


# In[6]:


can=np.zeros((300,300,3),dtype=np.uint8)


# In[14]:


def display(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(1)
    cv2.destroyWindow(name)


# In[7]:


centrex,centery=(can.shape[1]//2,can.shape[0]//2)
white=(255,255,255)
for radius in range(0,175,20):
    cv2.circle(can,(centrex,centery),radius,white)
display("circle",can)


# In[11]:


import cv2
image = cv2.imread('/Users/hammad/Desktop/hammad.jpeg')
radius = 400
center = (620, 500)
color = (255,255,0)
thickness = 15
image = cv2.circle(image, center, radius, color, thickness)
#cv2.imshow('Circle', image)
plt.imshow(image[:,:,::-1])


# In[55]:


img8 = cv2.imread('/Users/hammad/Desktop/hammad.jpeg')
def trans(img8,tx,ty):
    M=np.float64([[1,0,tx],[0,1,ty]])
    shift=cv2.warpAffine(img8,M,(img8.shape[1],img8.shape[0]))


# In[68]:


trans(img8,-100,-150)


# In[61]:


center=(img8.shape[1]//2,img8.shape[0]//2)
M=cv2.getRotationMatrix2D(center,45,1)
M


# In[65]:


rot=cv2.warpAffine(img8,M,(img8.shape[1],img8.shape[0]))
plt.imshow(rot[:,:,::-1])


# In[71]:


flip=cv2.flip(image,-1)
plt.imshow(flip[:,:,::-1])


# In[3]:


img11=cv2.imread("/Users/hammad/Desktop/image_03.jpeg")
img12=cv2.imread("/Users/hammad/Desktop/image_03.jpeg")
#both image have same size


# In[ ]:





# In[84]:


new=cv2.addWeighted(img11,0.9,img12,0.9,1)
plt.imshow(new[:,:,::-1])


# In[2]:


img11=cv2.imread("/Users/hammad/Desktop/image_03.jpeg")
def no(x):
    pass
cv2.namedWindow("Brightness control")
bright=cv2.createTrackbar("bright","Brightness control",75,255,no)
value=np.ones_like(img11,dtype="uint8")
while True:
    bright=cv2.getTrackbarPos("bright","Brightness control")
    bar=bright-127
    if bar >=0:
        value=np.ones_like(img11,dtype="uint8")*bar
        img_ctrl=cv2.add(img11,value)
    else:
        bright=127-bright
        value=np.ones_like(img11,dtype="uint8")*bright
        img_ctrl=cv2.subtract(img11,value)
    cv2.imshow("brightness",img_ctrl)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()    




# In[58]:


mask=np.zeros(img11.shape[:2],dtype="uint8")
cx,cy=mask.shape[1]//2,mask.shape[0]//2
arr=cv2.rectangle(mask,(cx-20,cy-20),(cx+20,cy+20),(255,255,255),-1)


# In[59]:


# cv2.imshow(arr[:,::-1])
arr
plt.imshow(arr)


# In[60]:


b = cv2.bitwise_xor(arr,arr)


# In[61]:


plt.imshow(b)


# In[70]:


mask=np.zeros(img11.shape[:2],dtype="uint8")
cx,cy=mask.shape[1]//2,mask.shape[0]//2
arr=cv2.rectangle(mask,(cx-100,cy-100),(cx+100,cy+100),(255,255,255),-1)
plt.imshow(arr)


# In[71]:


res= cv2.bitwise_and(img11,img11,mask=mask)
plt.imshow(res)


# In[49]:


mask=np.zeros_like(img11,dtype="uint8")
cx,cy=mask.shape[1]//2,mask.shape[0]//2
arr=cv2.circle(mask,(cx,cy),194,255,-1)
plt.imshow(arr)


# In[54]:


blur=cv2.blur(img11,(8,9))
plt.imshow(blur[:,:,::-1])


# In[ ]:


def face_detect(imgg):

    blob =  cv2.dnn.blobFromImage(imgg, 1,(300,300),(104,177,123),swapRB= True)
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()
    h,w = imgg.shape[:2]
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.40:
            # print(confidence)
            box= detections[0,0,i,3:7]*np.array([w,h,w,h])
            box=box.astype('int')
            pt1=(box[0],box[1])
            pt2=(box[2],box[3])
            cv2.rectangle(imgg,pt1,pt2, (0,255,0),1)
            text= 'score : {:.0f} %'.format(confidence*100)
            cv2.putText(imgg,text,pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    return imgg


# In[ ]:


cap = cv2.VideoCapture(0)
face_detection_model = cv2.dnn.readNetFromCaffe('./deploy.prototxt.txt',
                                './res10_300x300_ssd_iter_140000_fp16.caffemodel')
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    img_detection = face_detect(frame)

    cv2.imshow('face_detection',img_detection)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import json
import cv2
from PIL import Image,ImageDraw,ImageOps,ImageFont
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


image=Image.open('./imam id card.jpg')
img=image.copy()
img=ImageOps.exif_transpose(img)
f=json.load(open('./via_export_json.json'))
json_object = json.dumps(f, indent = 6)
  
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
def img_func(i,text):
    print(f['imam id card.jpg2702536']['regions'][i]['shape_attributes'])
    x1=f['imam id card.jpg2702536']['regions'][i]['shape_attributes']['x']
    y1=f['imam id card.jpg2702536']['regions'][i]['shape_attributes']['y']
    x2=f['imam id card.jpg2702536']['regions'][i]['shape_attributes']['width']
    y2=f['imam id card.jpg2702536']['regions'][i]['shape_attributes']['height']

    w, h = np.array(img).shape[0],np.array(img).shape[1]
    shape = [(x1,y1),(x1+x2,y1+y2)]
    img1 = ImageDraw.Draw(img)  
    font=ImageFont.truetype("arial.ttf",100)
    img1.text((x1+100,y1-100),text,(0,0,0),font=font)
    img.save('box_img.jpg')
    img1.rectangle(shape, outline ="green",width=30)
img_func(0,'country')
img_func(1,'name')


# In[ ]:


import cv2

import numpy as np

from matplotlib import pyplot as plt

img = cv2.imread('smarties.png')

img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernal =np.ones((5 ,5), np.float32)/25

dst = cv2.filter2D(img, -1, kernal)

blur = cv2.blur(img, (5,5))

gblur = cv2.GaussianBlur(img, (5,5), 0)

median = cv2.medianBlur(img, 5)

bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

titles = ['img', 'dst' ,'blur', "gblur", 'median', 'bilateralFilter']

images = [img,dst ,blur ,gblur,median,bilateralFilter]

for i in range(6):
    plt.subplot(2,3, i+1), plt.imshow(images[i], 'gray')

    plt.title(titles[i])
    
    plt.xticks([]), plt.yticks([])

    plt.show()


# In[ ]:


imp1 = Image.open("Landscape_2.jpeg")
imp2 = Image.open("Landscape_3.jpeg")
data1 = list(imp1.getdata())
data2 = list(imp2.getdata())
image_without_exif1 = Image.new(imp1.mode, imp1.size)
image_without_exif1.putdata(data1)
image_without_exif2 = Image.new(imp2.mode, imp2.size)
image_without_exif2.putdata(data2)
concat_img_pil = Image.new("RGB", (image_without_exif1.width + image_without_exif2.width, image_without_exif1.height),"white")
concat_img_pil.paste(image_without_exif1, (0,0))
concat_img_pil.paste(image_without_exif2, (image_without_exif1.width, 0))
plt.imshow(concat_img_pil)


# In[ ]:


from PIL.ExifTags import TAGS
i=Image.open("Landscape_3.jpeg")
exifdata = i.getexif()


# In[ ]:


for tag_id in exifdata:
    # get the tag name, instead of human unreadable tag id
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    # decode bytes 
    if isinstance(data, bytes):
        data = data.decode()
    print(f"{tag:25}: {data}")


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img11=cv2.imread("/Users/hammad/Desktop/image_03.jpeg")
mask=np.zeros_like(img11,dtype="uint8")
cx,cy=mask.shape[1]//2,mask.shape[0]//2
arr=cv2.circle(mask,(cx,cy),194,255,-1)
plt.imshow(arr[:,:,::-1])


# In[ ]:


import numpy as np
import cv2
img = cv2.imread("/Users/hammad/Desktop/images.png")
mask=np.array(img)
img=img.copy()
excluded_color = 139
indices_list = np.where(np.all(mask >= excluded_color, axis=-1))
mask[indices_list] = [0,0,255]
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(mask)


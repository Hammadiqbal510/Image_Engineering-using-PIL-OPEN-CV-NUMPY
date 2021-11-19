#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL  import Image
import matplotlib.pyplot as plt


# In[2]:


img1=Image.open("/Users/hammad/Downloads/as.jpeg")


# In[3]:


img1.show()


# In[4]:


plt.imshow(img1)


# In[5]:


img1.size


# In[6]:


img1.format


# In[7]:


img1.mode


# In[12]:


left=200
top=420
right=500
bottom=600
imgs=img1.crop((left,top,right,bottom))
plt.imshow(imgs)


# In[13]:


img2=img1.copy()


# In[14]:


img2


# In[15]:


left_r=img2.transpose(Image.FLIP_LEFT_RIGHT)
left_r.show()


# In[16]:


plt.imshow(left_r)


# In[17]:


left_r1=img2.transpose(Image.FLIP_TOP_BOTTOM)
plt.imshow(left_r1)


# In[18]:


left_r2=img2.transpose(Image.ROTATE_90)
#left_r3=img2.transpose(Image.ROTATE_270)
#left_r4=img2.transpose(Image.ROTATE_180)

plt.imshow(left_r2)


# In[33]:


new_size=(430,270)
size=img2.resize(new_size,Image.BOX)
#similralry other 6 interpolation technique is used in Image resizing
plt.imshow(size)


# In[35]:


angle=45
rotate=img2.rotate(angle)
plt.imshow(rotate)


# In[55]:


from PIL import ImageFont
from PIL import ImageDraw
img4=img2.copy()
water=ImageDraw.Draw(img4)
#font_type = ImageFont.truetype("arial.ttf", 18)
#draw = ImageDraw.Draw(img4)
#draw.text(xy=(120, 120), text= "download font you want to use", fill=(255,69,0), font=font_type)
#plt.imshow(img4)


# In[60]:


# font=ImageFont.truetype(font="msyhbd.ttf",size=100,encoding="unic")
font = ImageFont.load_default()
draw = ImageDraw.Draw(img4)
draw.text(( 20, 32), "Hammad", (255,0,0), font=font)
plt.imshow(img4)


# In[65]:


size=(300,400)
img5=img2.copy()
img5.thumbnail(size)
#the image you want to paste
img6=img2.copy()
img6.paste(img5,(0,1))
plt.imshow(img6)


# In[67]:


bg=img5.convert("L")


# In[68]:


bg


# In[69]:


plt.imshow(bg,cmap="gray")


# In[70]:


img7=img2.copy()


# In[73]:


format=img7.convert("HSV")
print(format.mode)
plt.imshow(format)


# In[75]:



import numpy as np
arr=np.array(img7)
arr


# In[76]:


arr.shape


# In[84]:


img_again=Image.fromarray(arr)
plt.imshow(arr)


# In[85]:


from PIL import ImageEnhance


# In[87]:


enhan=ImageEnhance.Color(img7).enhance(2.5)
plt.imshow(enhan)


# In[88]:


enhan=ImageEnhance.Contrast(img7).enhance(2.5)
plt.imshow(enhan)


# In[89]:


enhan=ImageEnhance.Brightness(img7).enhance(2.5)
plt.imshow(enhan)


# In[91]:


enhan=ImageEnhance.Sharpness(img7).enhance(5)
plt.imshow(enhan)


# In[106]:


# https://pythontic.com/image-processing/pillow/blend
# https://pillow.readthedocs.io/en/stable/reference/Image.html
img8=img2.copy()
img9=img8.copy()
img8.size
img7.size
#if their size is not equal sure that and resize it
img9=img9.resize(img8.size)
#blending
#use another image both are not same and remember it should be PNG format
alphaBlended2 = Image.blend(img8,img9, alpha=0.4)
plt.imshow(alphaBlended2)


# In[118]:


transform=img8.transform(img8.size,Image.AFFINE,(1,0.1,0.2*img8.size[0],0,1,0))
plt.imshow(transform)


# In[120]:


transform=img8.transform(img8.size,Image.EXTENT,(50,20,img8.size[0],img8.size[0]//1))
plt.imshow(transform)


# In[128]:


transform=img8.transform(img8.size,Image.EXTENT,(50,20,img8.size[0],img8.size[1]//3))
plt.imshow(transform)


# In[144]:


r,g,b=img8.split()
#its you choice which channel you want to put first
mer=Image.merge("RGB",(g,r,b))
plt.imshow(mer)


# In[176]:


"""#if YOu have all the files in dir and convert all images into png format 
#jpg top png and png to jpg 
from PIL import Image
import os
for f in os.listdir(r"."):
    if f.endswith(".png"):
        img=Image.open(f)
        fn,fext=os.path.splitext(f)
        img.save("png_images/{}.jpeg".format(fn))"""


# In[171]:


from PIL import Image
import os

directory = r'/Users/hammad/Desktop'
c=1
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        im = Image.open(filename)
        name='img'+str(c)+'.jpeg'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c+=1
        print(os.path.join(directory, filename))


# In[ ]:





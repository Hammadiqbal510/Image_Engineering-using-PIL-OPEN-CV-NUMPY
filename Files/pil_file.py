#!/usr/bin/env python
# coding: utf-8

# In[60]:


from PIL import Image,ImageOps
import imagehash
import matplotlib.pyplot


# In[61]:


img1=Image.open('../Landscape_2.jpeg')
img2=Image.open('../Landscape_3.jpeg')


# In[62]:


img1=ImageOps.exif_transpose(img1)
img2=ImageOps.exif_transpose(img2)


# In[63]:


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

get_concat_h(img1, img2).save('./newImg.jpg')


# In[64]:


con_img=Image.open('newImg.jpg')
hash=imagehash.average_hash(con_img)
print(hash)
plt.imshow(con_img )


# In[ ]:





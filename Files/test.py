#!/usr/bin/env python
# coding: utf-8

# In[31]:


import json
import cv2
from PIL import Image,ImageDraw,ImageOps,ImageFont
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


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





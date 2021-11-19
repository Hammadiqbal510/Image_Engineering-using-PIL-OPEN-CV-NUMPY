#!/usr/bin/env python
# coding: utf-8

# In[247]:


import json
import cv2
import matplotlib.pyplot as plt


# In[248]:


f = json.load(open('D:\JSON task\IMG20211102181143 (1).json'))
del f['textAnnotations'][0]
f


# In[249]:


fi = json.dumps(f,indent=4)
with open('new.json', 'w') as file:
    file.write(fi)


# In[250]:


img=cv2.imread('card.jpg')
img=img.copy()
img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
plt.imshow(img)


# In[252]:


plt.figure(figsize=(10,10))
for i in f['textAnnotations']:
    # print(i['description'],i['boundingPoly'])
    x1,y1=(i['boundingPoly']['vertices'][0]['x'],i['boundingPoly']['vertices'][0]['y'])
    x2,y2=(i['boundingPoly']['vertices'][2]['x'],i['boundingPoly']['vertices'][2]['y'])
    cv2.rectangle(img,(x2,y2),(x1,y1),(255,0,0),5)
    cv2.putText(img,i['description'],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,0),10)
plt.imshow(img)
img.shape


# In[ ]:





# In[ ]:





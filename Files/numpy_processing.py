#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img=plt.imread("/Users/hammad/Desktop/image_03.jpeg")
#print(type(img))
plt.imshow(img)


# In[7]:


red=img[:,:,0]
red


# In[8]:


img1=Image.fromarray(red)
img1.show()


# In[16]:


re=img[:,:,1]
re


# In[17]:


img2=Image.fromarray(re)
img2.show()


# In[24]:


img3=np.flipud(img)
img3


# In[25]:


img4=Image.fromarray(img3)
img4


# In[26]:


img4=np.fliplr(img)
img4


# In[27]:


img5=Image.fromarray(img4)
img5


# In[31]:


im_gray = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg').convert('L'))
plt.imshow(im_gray)


# In[45]:


from PIL import Image
import numpy as np

im = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg'))

im_R = im.copy()
im_R[:, :, (1, 2)] = 0
im_G = im.copy()
im_G[:, :, (0, 2)] = 0
im_B = im.copy()
im_B[:, :, (0, 1)] = 0

im_RGB = np.concatenate((im_R, im_G, im_B), axis=1)
# im_RGB = np.hstack((im_R, im_G, im_B))
# im_RGB = np.c_['1', im_R, im_G, im_B]

pil_img = Image.fromarray(im_RGB)
plt.imshow(pil_img)


# In[58]:


import numpy as np
from PIL import Image

im = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg').resize((256, 256)))
im_i = 255 - im
plot=Image.fromarray(im_i)
plt.imshow(im_i)


# In[59]:


import numpy as np
from PIL import Image

im = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg').resize((256, 256)))

im_32 = im // 32 * 32
im_128 = im // 128 * 128

im_dec = np.concatenate((im, im_32, im_128), axis=1)
plot=Image.fromarray(im_dec)
plt.imshow(im_dec)


# In[76]:


import numpy as np
from PIL import Image

im_black = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg').convert('L'))
Tresh=30
im_bool = im_black > Tresh
maxval = 255
im_bin = (im_black > Tresh) * maxval
plt.imshow(im_bin)


# In[99]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im_black = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg').convert("L"))
plt.imshow(im_black,cmap="gray")


# In[102]:


Tresh=120
im_bool = im_black > Tresh
maxval = 255
im_bin = (im_black > Tresh) * maxval
plt.imshow(im_bin)


# In[32]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img=plt.imread("/Users/hammad/Desktop/image_03.jpeg")
#print(type(img))
img=np.rot90(img,k=1, axes=(0,1))
plt.imshow(img)


# In[33]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img=plt.imread("/Users/hammad/Desktop/image_03.jpeg")
#print(type(img))
img=np.rot90(img,k=2)
plt.imshow(img)


# In[34]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img=plt.imread("/Users/hammad/Desktop/image_03.jpeg")
#print(type(img))
img=np.rot90(img,k=2, axes=(0,2))
plt.imshow(img)


# In[63]:


im = np.array(Image.open('/Users/hammad/Desktop/image_03.jpeg'))


# In[64]:


img=im[:,::-1,:]
plt.imshow(img)


# In[65]:


img=im[::-1,:,:]
plt.imshow(img)


# In[66]:


img=im[::-1,::-1,:]
plt.imshow(img)


# In[ ]:





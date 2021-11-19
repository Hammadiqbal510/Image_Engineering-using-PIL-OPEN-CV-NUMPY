#!/usr/bin/env python
# coding: utf-8

# In[135]:


import subprocess


# In[141]:


image='../Landscape_3.jpeg'
exe='hachoir-metadata'


# In[142]:


process=subprocess.Popen([exe,image],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
data=process.stdout
for i in data:
    print(i.strip())


# In[139]:


from PIL import Image
from PIL.ExifTags import TAGS

def get_exif():
    i=Image.open('../Landscape_3.jpeg')
    info = i.getexif()
    return {TAGS.get(tag): value for tag, value in info.items()}

print(get_exif())


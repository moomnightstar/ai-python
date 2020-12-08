
# -*- coding: UTF-8 -*-

import sys
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

image_path=str(sys.argv[1])
k=int(sys.argv[2])
print('k=' + str(k))

im = Image.open(image_path)
print(im.mode,im.size,im.format)
#RGB　８-bit 0~255
#pix = im.load()

width = int(im.size[0])  #图片的宽
height = int(im.size[1]) #图片的高

A= np.array(im)
print(A.shape)
#转换成二维数组
B=np.reshape(A,(-1,3))
print(B.shape)
print(B)
#聚类开始
kmeans = KMeans(n_clusters=k, random_state=0).fit(B)
cluster_index=kmeans.predict(B)
print(kmeans.cluster_centers_)
#聚类结束

im_new=Image.new("RGB",(width,height))
for x in range(height):
    for y in range(width):
        r,g,b= kmeans.cluster_centers_[cluster_index[x*width+y]][0:3]
        im_new.putpixel((y,x),(int(r),int(g),int(b)))
im_new.show()
result_pic=image_path[:-4]+'_k='+str(k)+'.jpg'
im_new.save(result_pic)
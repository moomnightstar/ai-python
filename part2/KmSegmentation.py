
# -*- coding: UTF-8 -*-
import random
import sys
from PIL import Image

image_path=str(sys.argv[1])
k=int(sys.argv[2])
print('k=' + str(k))

im = Image.open(image_path)
print(im.mode,im.size,im.format)
#RGB　８-bit 0~255
pix = im.load()
width = int(im.size[0])  #图片的宽
height = int(im.size[1]) #图片的高

#设定簇数目
k_class=[[0 for i in range(3)] for i in range(k)]

pix_class=[[0 for i in range(height) ]for i in range(width)]

for x in range(k):
    for y in range(3):
        k_class[x][y]=random.randrange(0,255,1)

print(k_class)
flag=1
while flag==1:
    next_k_class = [[0 for i in range(3)] for i in range(k)]
    k_class_num = [0 for i in range(k)]
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y]
            dist = -1
            for i in range(k):#计算每个点离那个中心点最近
                tmp1 = (r - k_class[i][0]) * (r - k_class[i][0])
                tmp2 = (g - k_class[i][1]) * (g - k_class[i][1])
                tmp3 = (b - k_class[i][2]) * (b - k_class[i][2])
                if dist == -1:
                    pix_class[x][y] = i
                    dist = tmp1 + tmp2 + tmp3
                elif tmp1 + tmp2 + tmp3 < dist:
                    pix_class[x][y] = i
                    dist = tmp1 + tmp2 + tmp3
            next_k_class[pix_class[x][y]][0] += r
            next_k_class[pix_class[x][y]][1] += g
            next_k_class[pix_class[x][y]][2] += b
            k_class_num[pix_class[x][y]] += 1
    for x in range(k):
        for y in range(3):
            if k_class_num[x] != 0:
                next_k_class[x][y] = next_k_class[x][y] // k_class_num[x]#重新计算中心点，这里取平均值
    print(next_k_class)
    flag=0
    for x in range(k):
        if next_k_class[x][0:3] != k_class[x][0:3]: #还有类未固定
            flag = 1
            break
    for x in range(k):
        k_class[x][0:3] = next_k_class[x][0:3]
#聚类结束
im_new=Image.new("RGB",(width,height))
for x in range(width):
    for y in range(height):
        r,g,b= k_class[pix_class[x][y]][0:3]
        im_new.putpixel((x,y),(r,g,b))
im_new.show()
result_pic='result_picture_k='+str(k)+'.jpg'
im_new.save(result_pic)
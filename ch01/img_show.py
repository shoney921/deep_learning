# coding: utf-8
import os, matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread(os.path.dirname(__file__)
             + '/../dataset/cactus.png') # 이미지 읽어오기
plt.imshow(img)

plt.show()

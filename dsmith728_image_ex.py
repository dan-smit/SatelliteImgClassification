from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('./data/valid/images/023_jpg.rf.7c92f2c48e55b738e814be3883856016.jpg') #(640x640) 
img2 = Image.open('./data/train/images/004_jpg.rf.3d4088be97f761ef34881054e8321d9b.jpg') #(640x640)
imglabel = Image.open('ImgALabels.jpg') 
img2label = Image.open('ImgBLabels.jpg') 

img_array = np.array(img)
img2_array = np.array(img2)
imglabel_array = np.array(imglabel)
img2label_array = np.array(img2label)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.imshow(img_array)
ax2.imshow(img2_array)
ax3.imshow(imglabel_array)
ax4.imshow(img2label_array)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()

ax1.set_title('Image A')
ax2.set_title('Image B')
ax3.set_title('Image A Labels')
ax4.set_title('Image B Labels')

fig.tight_layout()
plt.show()
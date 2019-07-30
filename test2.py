from PIL import Image, ImageSequence
import numpy as np
im1 = Image.open('./png/105.gif')
iter_im1 = list(ImageSequence.Iterator(im1))
im2 = Image.open('./png/140.gif')
iter_im2 = list(ImageSequence.Iterator(im2))

print(len(iter_im1))
for i in range(len(iter_im1)):
    conv1 = iter_im1[i].convert('RGB')
    conv2 = iter_im2[i].convert('RGB')
    aaa = np.array(conv1)
    bbb = np.array(conv2)
    print(np.sum())

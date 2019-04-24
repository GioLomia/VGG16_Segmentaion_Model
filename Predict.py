import glob
import os
import random

import cv2
import numpy as np

import LoadBatches
import VGGSegnet

n_classes = 10
model_name = 'vgg_segnet'
images_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/pred_images"
output_path = "C:/Users/lomiag/PycharmProjects/Image_Object_Tranfer/Preds"
input_width = 224
input_height = 224
epoch_number = 1

modelFns = {'vgg_segnet': VGGSegnet.VGGSegnet}
modelFN = modelFns[model_name]

save_weights_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Weights/Weights"

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(save_weights_path + "." + str(epoch_number))
# m.load_weights( "C:/Users/lomiag/Desktop/Object Transfer Project/Weights/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz" )
m.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

lister = os.listdir(images_path)
print(images_path)
new_im_dir = []
for i in lister:
    print(images_path + "/" + i)
    new_im_dir.append(images_path + "/" + i)

print(new_im_dir)

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images = new_im_dir
images.sort()
print(images)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

for imgName in images:

    outName = imgName.replace(images_path, output_path)
    X = LoadBatches.getImageArr(imgName, input_width, input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
    print(outName)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_weights_path", type=str)
# parser.add_argument("--epoch_number", type=int, default=5)
# parser.add_argument("--test_images", type=str, default="")
# parser.add_argument("--output_path", type=str, default="")
# parser.add_argument("--input_height", type=int, default=224)
# parser.add_argument("--input_width", type=int, default=224)
# parser.add_argument("--model_name", type=str, default="")
# parser.add_argument("--n_classes", type=int)
# args = parser.parse_args()
#
# n_classes = args.n_classes
# model_name = args.model_name
# images_path = args.test_images
# input_width = args.input_width
# input_height = args.input_height
# epoch_number = args.epoch_number


"""
evaluate the model
"""

import Models
import LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

n_classes = 12
model_name = "fcn8"
images_path = "/home/ceo1207/Datasets/dataset1/images_prepped_test"
input_height = 352
input_width = 480
modelFns = {
    'vgg_segnet': Models.VGGSegnet.VGGSegnet,
    'vgg_unet': Models.VGGUnet.VGGUnet,
    'vgg_unet2': Models.VGGUnet.VGGUnet2,
    'fcn8': Models.FCN8.FCN8,
    'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights("model_1.h5")

images = sorted(
    glob.glob(
        os.path.join(images_path, "*.jpg")) +
    glob.glob(
        os.path.join(images_path, "*.png")) +
    glob.glob(
        os.path.join(images_path, "*.jpeg")))

colors = [(random.randint(0, 255),
           random.randint(0, 255),
           random.randint(0, 255)) for _ in range(n_classes)]

for imgName in images:
    X = LoadBatches.getImageArr(imgName, input_width, input_height)
    pr = m.predict(np.array([X]))
    pr = np.squeeze(pr)
    pr = np.argmax(pr, axis=2)
    seg_img = np.zeros((pr.shape[0], pr.shape[1], 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    # seg_img = cv2.resize(seg_img, (input_width, input_height))
    seg_img = seg_img.astype("int")
    plt.imshow(seg_img)
    plt.show()

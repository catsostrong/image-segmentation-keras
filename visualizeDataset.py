
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import argparse
import os

def imageSegmentationGenerator(images_path, segs_path, n_classes):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(
        glob.glob(
            images_path +
            "*.jpg") +
        glob.glob(
            images_path +
            "*.png") +
        glob.glob(
            images_path +
            "*.jpeg"))
    segmentations = glob.glob(
        segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    # get a color for different classes
    colors = [
        (random.randint(
            0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
        if os.path.isfile(im_fn) and os.path.isfile(seg_fn):
            assert(im_fn.split('/')[-1] == seg_fn.split('/')[-1])

            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            print np.unique(seg)

            seg_img = np.zeros_like(seg)

            for c in range(n_classes):
                seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                                     (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                                     (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                                     (colors[c][2])).astype('uint8')

            fig = plt.figure(figsize=(10, 5))
            x = fig.add_subplot(1, 2, 1)
            x.imshow(img)
            y = fig.add_subplot(1, 2, 2)
            y.imshow(seg_img)
            plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str)
parser.add_argument("--annotations", type=str)
parser.add_argument("--n_classes", type=int)
args = parser.parse_args()
imageSegmentationGenerator(args.images, args.annotations, args.n_classes)


"""
usage

python visualizeDataset.py 
 --images= "/home/ceo1207/Datasets/dataset1/images_prepped_train" --annotations="/home/ceo1207/Datasets/dataset1/annotations_prepped_train"  --n_classes=10 
 
"""
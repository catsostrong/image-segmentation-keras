import numpy as np
import cv2

def getImageArr(path, width, height):

    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = img.astype("float")
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img/255.0

    # if imgNorm == "sub_and_divide":
    #     img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    # elif imgNorm == "sub_mean":
    #     img = cv2.resize(img, (width, height))
    #     img = img.astype(np.float32)
    #     img[:, :, 0] -= 103.939
    #     img[:, :, 1] -= 116.779
    #     img[:, :, 2] -= 123.68
    # elif imgNorm == "divide":
    #     img = cv2.resize(img, (width, height))
    #     img = img.astype(np.float32)
    #     img = img / 255.0

    return img



from keras.utils import to_categorical
def getSegmentationArr(path, nClasses, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]
    label = to_categorical(img, num_classes=nClasses)

    # seg_labels = np.zeros((height, width, nClasses))
    # img = cv2.imread(path, 1)
    # img = cv2.resize(img, (width, height))
    # img = img[:, :, 0]
    #
    # for c in range(nClasses):
    #     seg_labels[:, :, c] = (img == c).astype(int)
    # seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return label

import os

# generator a batch for training endlessly
def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width):
    generator = one_batch_Generator(images_path, segs_path, n_classes, input_height, input_width)
    while True:
        X = np.zeros((batch_size, input_height, input_width, 3))
        Y = np.zeros((batch_size, input_height, input_width, n_classes))
        for index_batch in range(batch_size):
            tmp_x, tmp_y = generator.next()
            X[index_batch] = tmp_x
            Y[index_batch] = tmp_y
        yield X, Y


def one_batch_Generator(images_path, segs_path, n_classes, input_height, input_width):
    img_list = os.listdir(images_path)
    steps = len(img_list)
    print "hava ", steps, " images"
    while True:
        np.random.shuffle(img_list)
        for item in img_list:
            im = os.path.join(images_path, item)
            seg = os.path.join(segs_path, item)
            if os.path.isfile(im) and os.path.isfile(seg):
                x = getImageArr(im, input_width, input_height)
                y = getSegmentationArr(
                        seg,
                        n_classes,
                    input_width, input_height)
                yield x,y




# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight


from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
import os




def FCN8(nClasses, input_height=360, input_width=480):
    vgg = VGG16()
    vgg = Model(vgg.input, vgg.get_layer(name="block5_pool").output)
    vgg.summary()
    f5 = vgg.get_layer(name="block5_pool").output
    f4 = vgg.get_layer(name="block4_pool").output
    f3 = vgg.get_layer(name="block3_pool").output

    y_1 = Conv2D(4096, (7, 7), activation='relu', padding='same')(f5)
    y_1 = Dropout(0.5)(y_1)
    y_1 = Conv2D(4096, (1, 1), activation='relu', padding='same')(y_1)
    y_1 = Dropout(0.5)(y_1)
    y_1 = Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', padding='same')(y_1)
    # scale for 2
    y_1 = Conv2DTranspose(
        nClasses, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding="same")(y_1)

    y_2 = Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', padding="same")(f4)

    y_2 = Add()([y_1, y_2])

    y_2 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding="same")(y_2)

    y_3 = Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', padding='same')(f3)
    y_3 = Add()([y_3, y_2])

    y_overall = Conv2DTranspose(nClasses, kernel_size=(16, 16), strides=(8, 8),
                                use_bias=False, padding="same", activation="softmax")(y_3)

    vgg = Model(vgg.input, y_overall)
    vgg.summary()
    plot_model(vgg, show_shapes=True, to_file='model8s_1.png')

    # useless
    # vgg.inputs = [Input(shape=(input_height, input_width, 3))]
    # useless
    # vgg.layers[0] = InputLayer(input_shape=(input_height, input_width, 3))

    x = Input(shape=(input_height, input_width, 3))
    y = vgg(x)
    vgg = Model(x, y)
    # model test
    # a = np.ones((1,352,480,3))
    # b = vgg.predict(a)
    # print b.shape

    # another way to change Input shape presevering the structure of the network

    return vgg


if __name__ == '__main__':

    m = FCN8(11)

    plot_model(m, show_shapes=True, to_file='model8s.png')

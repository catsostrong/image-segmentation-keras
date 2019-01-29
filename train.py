
import Models
import LoadBatches
import os
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_weights_path", type = str  )
# parser.add_argument("--train_images", type = str  )
# parser.add_argument("--train_annotations", type = str  )
# parser.add_argument("--n_classes", type=int )
# parser.add_argument("--input_height", type=int , default = 224  )
# parser.add_argument("--input_width", type=int , default = 224 )
#
# parser.add_argument('--validate',action='store_false')
# parser.add_argument("--val_images", type = str , default = "")
# parser.add_argument("--val_annotations", type = str , default = "")
#
# parser.add_argument("--epochs", type = int, default = 5 )
# parser.add_argument("--batch_size", type = int, default = 2 )
# parser.add_argument("--val_batch_size", type = int, default = 2 )
# parser.add_argument("--load_weights", type = str , default = "")
#
# parser.add_argument("--model_name", type = str , default = "")
# parser.add_argument("--optimizer_name", type = str , default = "adadelta")
#
#
# args = parser.parse_args()
#
# train_images_path = args.train_images
# train_segs_path = args.train_annotations
# train_batch_size = args.batch_size
# n_classes = args.n_classes
# input_height = args.input_height
# input_width = args.input_width
# validate = args.validate
# save_weights_path = args.save_weights_path
# epochs = args.epochs
# load_weights = args.load_weights
#
# optimizer_name = args.optimizer_name
# model_name = args.model_name




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 367 images
train_images_path = "/home/ceo1207/Datasets/dataset1/images_prepped_train"
# 360, 480, 3
train_segs_path = "/home/ceo1207/Datasets/dataset1/annotations_prepped_train"
train_batch_size = 1
n_classes = 12    # 11+1 background
# should be even number
input_height = 352
input_width = 480

save_weights_path = "model"
epochs = 2
load_weights = None

model_name = "fcn8"


modelFns = {
    'vgg_segnet': Models.VGGSegnet.VGGSegnet,
    'vgg_unet': Models.VGGUnet.VGGUnet,
    'vgg_unet2': Models.VGGUnet.VGGUnet2,
    'fcn8': Models.FCN8.FCN8,
    'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy',
          optimizer="adam",
          metrics=['accuracy'])


if load_weights:
    m.load_weights(load_weights)



train_generator = LoadBatches.imageSegmentationGenerator(
    train_images_path,
    train_segs_path,
    train_batch_size,
    n_classes,
    input_height,
    input_width)


from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="log")
for ep in range(50):
    m.fit_generator(
        train_generator, epochs=1, steps_per_epoch=360, callbacks=[tensorboard])
    m.save(os.path.join(save_weights_path, "{}.h5".format(ep)))

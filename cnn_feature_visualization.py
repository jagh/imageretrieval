"""
+ Using a VGG19 Network to generate a heatmap features of an input imageself.
+ Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
+ Script base in the binary classifier kernel:
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input
import numpy as np
import keras.backend as K

from keras.applications.vgg19 import VGG19, decode_predictions
# from keras.applications.densenet import DenseNet121, decode_predictions

from PIL import Image
from IPython.display import display
from glob import glob


## Set Dataset path
dir_Dset = "cxr_dataset/"
train_data_dir = str(dir_Dset + "/train/")
validation_data_dir = str(dir_Dset + "/val/")
test_data_dir = str(dir_Dset + "/test/")

## prevent OOM issues
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

## returns the i:th layer's activations of model
def get_activations(i, model):
    return K.function([model.layers[0].input], [model.layers[i].output])

## so we can disable all prints at once
## but still get detailed info if we want to debug
def dprint(*args, debug):
    if debug:
        print(args)

## shows the activation heat map of the input image
def show_heatmap(name_img, inp_img, model, index, alpha=0.7, debug=False):
    # convert the image into a numpy array
    inp_arr = np.array(inp_img).reshape(1, inp_img.width, inp_img.height, 3)
    # predict the class of the image and print the top 3 predictions
    pred = model.predict([inp_arr])
    print([(label, conf) for _, label, conf in decode_predictions(pred)[0][:3]])

    # fetch the activations of layer index
    out = get_activations(index, model)([inp_arr])[0][0]
    dprint("activations", out.shape, debug=debug)

    # for each region of the activation map, calculate the average filter activations
    out_avg = np.mean(out, -1)
    dprint("post avg", out_avg.shape, debug=debug)

    # repeat the array into 3 dimensions
    out_avg = np.repeat(out_avg[:, :, np.newaxis], 3, axis=2)
    dprint("post repeat", out_avg.shape, debug=debug)

    # normalize the values into the range [0,1]
    dprint("pre normalize", np.amin(out_avg), np.amax(out_avg), debug=debug)
    out_avg /= np.amax(out_avg)
    dprint("post normalize", np.amin(out_avg), np.amax(out_avg), debug=debug)

    # transform the values into RGB range with a pink tint
    out_avg *= (255,0,128)
    dprint("post denormalize", np.amin(out_avg), np.amax(out_avg), debug=debug)

    # convert the average activations into an image and resize it to the input shape
    heatmap = Image.fromarray(np.uint8(out_avg))
    heatmap = heatmap.resize((inp_img.width, inp_img.height), Image.BICUBIC)

    # superimpose the heatmap on top of the input image
    input_heatmap = Image.blend(inp_img, heatmap, alpha)

    # show the result
    display(input_heatmap)
    heatmap.save(str("cxr_dataset/visualization_samples/"+name_img+"-heatmap.bmp"))
    input_heatmap.save(str("cxr_dataset/visualization_samples/"+name_img+"-heatmap_CXR.bmp"))



## use VGG19 with pretrained ImageNet weights
vgg = VGG19()

## Take activations from the last MaxPool layer
activations_index = -5
assert "block5_pool" == vgg.layers[activations_index].name
vgg.summary()


def load_images(path, img_w=224, img_h=224):
    paths = glob(path)
    for p in paths:
        inp_img = Image.open(p)
        yield inp_img.resize((img_w, img_h))

name_img = "p01-normal"
for img in load_images(str("cxr_dataset/visualization_samples/"+name_img+".png")):
    show_heatmap(name_img, img, vgg, index=activations_index, alpha=0.8, debug=False)

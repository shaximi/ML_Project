from tensorflow.keras import models
from tf_explain.callbacks.grad_cam import GradCAM
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tools for accessing and reading data (you do not need to use all of them)
import os
import pathlib
import PIL
import cv2
import skimage 
from IPython.display import Image, display
from matplotlib.image import imread
import matplotlib.cm as cm

# Tensorflow basics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

'''New Grad-CAM Image Explain'''




# inet_model.layers[-1].activation = None # Remove last layer's softmax
# print(inet_model.summary())
# tf.keras.utils.plot_model(inet_model, show_shapes=True, show_layer_names=True, expand_nested=True)


    
# print("Done")

# img_path = 'D:/shen/d/2class/k-ford_test/256a_v1_gc/tr/bad/1.png'
# display(Image(img_path))



def get_img_array(img_path,size):
    # `Sİze of image 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=(256,256))
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

last_conv_layer_name = "block5_conv2"

inet_model = tf.keras.models.load_model('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81vgg.h5')
print(inet_model.summary())
df_test = pd.read_csv('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81test_dataset.csv')


for i in range(len(df_test)):
    if df_test.iat[i,1]=='bad' and df_test.iat[i,2]==0 or df_test.iat[i,1]=='good' and df_test.iat[i,2]==1:
        path_str=df_test.iat[i,0]
        img_array = get_img_array(path_str, size=256)
    
    
    
        # Generate class activation heatmap
        # heatmap = make_gradcam_heatmap(img_array, grad_cam_model, last_conv_layer_name)
        heatmap = make_gradcam_heatmap(img_array, inet_model, last_conv_layer_name)
    
    
        # Display heatmap
        plt.matshow(heatmap)
        plt.show()
        save_path=path_str
        save_and_display_gradcam(path_str, heatmap,save_path)
    print(i)
























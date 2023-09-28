import os
import keras
from keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import numpy as np
'''LIME Image Explain From Internet'''
'''https://github.com/prodramp/DeepWorks/tree/main/MLI-XAI'''

print('Notebook run using keras:', keras.__version__)
first=True



def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)



inet_model = tf.keras.models.load_model('D:/shen/d/2class/code/2dense_VGG16_train_model_batchsize_hot8.h5')
images = transform_img_fn([os.path.join('D:/shen/d/2class/LIME/256a_v1/bad/','31.png')])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
preds = inet_model.predict(images)




import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

explainer = lime_image.LimeImageExplainer()



# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=100)

explanation

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

#########合成特徵熱力圖############
if first==True:  #第一次清空歸零
    full= np.empty((256,256), dtype=int)
    for i in range(0,256):#列
        for j in range(0,256):#欄
            full[j,i]=0            

for i in range(0,256):#列
    for j in range(0,256):#欄
        del fu;del ma
        fu=full[j,i]
        ma=mask[j,i]
        del te
        if ma>0:
            te=int(fu)+int(ma)
        else:
            te=int(fu)
        full[j,i]=te
        
first=False


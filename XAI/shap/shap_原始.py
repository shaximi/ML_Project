import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Sequential, layers
from keras import models
import keras.backend as K
from keras.preprocessing import image
import shap
'''SHAP Image Explain From Internet'''
'''https://www.kaggle.com/code/rambierestelle/unfolding-cnn-layers-visuals-and-shap'''

np.random.seed(1)
## Batch parameters
BATCH_SIZE = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256

list_files = []
list_category = []
train_dir = 'D:/shen/d/2class/shap/256a_v1'
# for dirname, _, filenames in os.walk(train_dir):
#     for filename in filenames:
#         list_files.append(os.path.join(dirname, filename))
#         list_category.append(dirname.split('/')[7])

training_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir+'/train',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE

)

valid_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir+'/valid',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

classes = training_set.class_names
####################################

fig,ax=plt.subplots(4,3)
fig.set_size_inches(15,15)
for next_element in valid_set:
  x_batch, y_batch = next_element
  for i in range (0,4):
    for j in range(3):
      random_example = np.random.randint(0, BATCH_SIZE)
      ax[i,j].imshow(x_batch[random_example]/250)
      ax[i,j].set_title('Status: '+ classes[y_batch[random_example].numpy()])
    break

####################################


from tensorflow.keras.layers import Dropout,Flatten,Dense,BatchNormalization

def create_model():
    
    # load pretrained model 'VGG16'
    base_model=tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224 ,3))
    trainable = False
    base_model.trainable = False
      
    model=Sequential()
    model.add(base_model)
    model.add(Dropout(0.3))           # dropout for preventing overfitting
    model.add(Flatten())
    model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
    # model.add(Dropout(0.3))
    model.add(Dense(2,activation='softmax',kernel_initializer='glorot_normal'))    # softmax classification for 5 labels
      
    


    return model
cnn_model = create_model()
cnn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              mode='max',
                                              patience=2,
                                         restore_best_weights=True)

################################
img = tf.keras.preprocessing.image.load_img("D:/shen/d/2class/shap/256a_v1/train/bad/991.jpg", target_size=(256, 256))
img = tf.keras.preprocessing.image.img_to_array(img)
# img = img.astype('int32')

img1 = img[None,:,:,:]

get_train_kpi = cnn_model.fit(training_set,
                validation_data=valid_set,
                epochs=1,)
cnn_model.save('111111.h5')
#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
cnn_model = tf.keras.models.load_model('D:/shen/d/2class/code/2dense_VGG16_train_model_batchsize16.h5')


explainer = shap.DeepExplainer(cnn_model,x_batch.numpy())
shap_values = explainer.shap_values(x_batch.numpy()[1].reshape(1,256,256,3), ranked_outputs=1, check_additivity=False)
shap.image_plot(shap_values[0],x_batch.numpy()[1].reshape(1,256,256,3), show=False)






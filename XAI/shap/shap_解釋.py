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
import pandas as pd
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
from tensorflow.keras.preprocessing import image
import shap
import time
import os,sys
'''SHAP Image Explain'''


print('Notebook run using keras:', keras.__version__)
first=True


cnn_model = tf.keras.models.load_model('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81vgg.h5')

IMG_HEIGHT = 256
IMG_WIDTH = 256

#shap讀照片比較麻煩
test_set = tf.keras.preprocessing.image_dataset_from_directory(
    'D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/sh/',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False,
    batch_size=200)

print(test_set.class_names)
####################################


for next_element in test_set:
  x_batch, y_batch = next_element

explainer = shap.DeepExplainer(cnn_model,x_batch.numpy())
for i in range(len(x_batch)):


    # img = tf.keras.preprocessing.image.load_img("", target_size=(256, 256))
    # img = tf.keras.preprocessing.image.img_to_array(img)
    # img1 = img[None,:,:,:]

    print(i)
    shap_values = explainer.shap_values(x_batch.numpy()[i].reshape(1,256,256,3), ranked_outputs=1, check_additivity=False)
    shap.image_plot(shap_values[0],x_batch.numpy()[i].reshape(1,256,256,3), show=False)
    time.sleep(10) #sleep 10s 防止CPU過熱
    
    
    
###########
#此而外程式不必執行，功能為將csv中預測正確的照片連結 另存成圖片
import time
import cv2
import pandas as pd
df_test = pd.read_csv('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81test_dataset.csv')


j=0
for i in range(len(df_test)):
    if df_test.iat[i,1]=='bad' and df_test.iat[i,2]==0:
        path_str=df_test.iat[i,0]
        print(path_str)
        j+=1
        img1 = cv2.imread(path_str)
        cv2.imwrite(path_str, img1)
    if df_test.iat[i,1]=='good' and df_test.iat[i,2]==1:
        path_str=df_test.iat[i,0]
        print(path_str)
        j+=1
        img1 = cv2.imread(path_str)
        cv2.imwrite(path_str, img1)
print('right num =>',j)

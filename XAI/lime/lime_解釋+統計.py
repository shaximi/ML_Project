import os
import keras
from keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

'''LIME Image Explain & heat map counting'''

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




# images = transform_img_fn([os.path.join('D:/shen/d/2class/LIME/256a_v1/bad/','31.png')])
# # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
# plt.imshow(images[0] / 2 + 0.5)
# preds = inet_model.predict(images)



inet_model = tf.keras.models.load_model('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81vgg.h5')
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
    
    
    
from lime import lime_image


df_test = pd.read_csv('D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/5_81test_dataset.csv')#讀取照片

import time

for i in range(len(df_test)):
    if df_test.iat[i,1]=='bad' and df_test.iat[i,2]==0 or df_test.iat[i,1]=='good' and df_test.iat[i,2]==1:#過濾預測正確案例

        path_str=df_test.iat[i,0]
        print(path_str)
        images = transform_img_fn([os.path.join(path_str)])
    
    
        explainer = lime_image.LimeImageExplainer()
        # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
        explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=100)
        
        explanation
        
        from skimage.segmentation import mark_boundaries
        '''效果1'''
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        # plt.show()
        # time.sleep(5)
        '''效果2'''
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        # plt.show()
        # time.sleep(5)
        '''效果3'''
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=8, hide_rest=False)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
        plt.imsave(path_str,mark_boundaries(temp / 2 + 0.5, mask))
        time.sleep(8) #sleep 8s 防止CPU過熱

##########統計Heatmap###########

        if first==True:
            full= np.empty((256,256), dtype=int)
            for i in range(0,256):#列
                for j in range(0,256):#欄
                    full[j,i]=0
        for i in range(0,256):#列
            for j in range(0,256):#欄
                fu=0;ma=0;
                del fu;del ma
                fu=full[j,i]
                ma=mask[j,i]
                te=0;
                del te
                if ma>0:
                    te=int(fu)+int(ma)
                else:
                    te=int(fu)
                full[j,i]=te
        first=False
    print(i)
    
#########合成特徵熱力圖############
for x in range(20):#列
    fulll= np.empty((256,256), dtype=int)
    for i in range(0,256):#列
        for j in range(0,256):#欄
            fulll[j,i]=full[j,i]#複製到fulll做處理
    for i in range(0,256):#列
        for j in range(0,256):#欄
            fu=0;ma=0;
            del fu;del ma
            fu=fulll[j,i]
            if fu>x:
                fulll[j,i]=1
            else:
                fulll[j,i]=0

    plt.imshow(fulll)
    plt.show()
    plt.imsave("lime"+str(x)+".png",fulll)

#####################################################################


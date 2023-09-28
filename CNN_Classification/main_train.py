import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score,recall_score
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
# local_device_protos = device_lib.list_local_devices()
# [print(x) for x in local_device_protos if x.device_type == 'GPU']
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)
# print(tf.test.is_gpu_available())
from tensorflow.keras.applications import VGG16,DenseNet121,ResNet50,Xception,InceptionV3,EfficientNetB3
from tensorflow.keras import layers, Model, utils
######################################################
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, train_test_split
import pathlib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout,Flatten,Dense,BatchNormalization




na='VGG16_02_b=0.85_r255_z256_over'
# data_path = pathlib.Path(r'D:/shen/d/2class/k-ford_test/256n_v1_final/tr/neck')
# data_path = pathlib.Path(r'D:/shen/d/2class/k-ford_test/256a_v1_final/tr1/tr')
data_path = pathlib.Path(r'D:/shen/d/2class/k-ford_test/156a_v1_final/tr2')
# glob all 'jpg' image files
img_path = list(data_path.glob('**/*.png'))
# split label names from file directory
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path))

pd_img_path = pd.Series(img_path, name='PATH').astype(str)
pd_img_labels = pd.Series(img_labels, name='LABELS').astype(str)
img_df = pd.merge(pd_img_path, pd_img_labels, right_index=True, left_index=True)
img_df = img_df.sample(frac = 1).reset_index(drop=True)
img_df.head()

img_df['LABELS'].value_counts(ascending=True)
# It is small dataset

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i+1)
  plt.imshow(plt.imread(img_df.PATH[i]))
  plt.title(img_df.LABELS[i])
  plt.axis("off")
  
  
img_df_over, test_dataset = train_test_split(img_df, train_size=0.85,shuffle=False,random_state=None,stratify=None)
print("Number of train data:", img_df_over.shape[0])
print("Number of test data:", test_dataset.shape[0])
  
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
  
# 創建 RandomOverSampler 物件
oversampler = RandomOverSampler(sampling_strategy=1)
# 對樣本進行過採樣
X_over, y_over = oversampler.fit_resample(img_df_over['PATH'].values.reshape(-1, 1), img_df_over['LABELS'])

# 將過採樣後的資料轉換為 DataFrame
img_df_over = pd.DataFrame({'PATH': X_over.flatten(), 'LABELS': y_over})

# 重新隨機排列資料
img_df_over = img_df_over.sample(frac = 1).reset_index(drop=True)
# 計算每個類別的樣本數量
class_counts = img_df_over['LABELS'].value_counts()

# 計算每個類別的樣本比例
class_proportions = class_counts / class_counts.sum()

print("平衡後資料比例:" ,class_proportions) 
  


# resize image to (255,255)
width = 256
height = 256

# use tensorflow real-time image data augmentation
t_datagen = ImageDataGenerator(rescale=1/255.0,         # [0,255] -> [0,1]
                     horizontal_flip = False,  # chess pieces look simillar horizontally
                     rotation_range = 2,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range = 0.3,
                     #validation_split=0.2
                     )

v_datagen = ImageDataGenerator(rescale=1/255.0,        # [0,255] -> [0,1]
                     horizontal_flip = False)  # chess pieces look simillar horizontally


test_gen = ImageDataGenerator(rescale=1/255.0,
                      horizontal_flip = False) # just rescaling for test data
test_ds = test_gen.flow_from_dataframe(test_dataset, x_col='PATH', y_col='LABELS',
                             target_size=(width,height),
                             class_mode = 'categorical', 
                             color_mode = 'rgb',
                             batch_size = 16, shuffle = False)






def create_model():  #此建模方式無法在之後詳細列出模型層列
    # load pretrained model 'VGG16'
    base_model=keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(width, height ,3))
    trainable = False
    base_model.trainable = False
      
    model=Sequential()
    model.add(base_model)
    model.add(BatchNormalization())   # batch normalization
    # model.add(Dropout(0.3))           # dropout for preventing overfitting
    model.add(Flatten())
    model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
    # model.add(Dropout(0.3))
    model.add(Dense(2,activation='softmax',kernel_initializer='glorot_normal'))    # softmax classification for 5 labels
     
    model.compile(optimizer=Adam(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    print(model.summary())
    return model

def create_model_new():#此建模方式可在之後詳細列出模型層列

    # load pretrained model 'VGG16'
    base_model=VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(width, height ,3), input_tensor=None)
    
    x = base_model.output
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='linear')(x)
    x = Flatten()(x)
    # x = layers.BatchNormalization()(x)  #shap中無法使用
    # x = layers.Dropout(0.2)(x)
    output_layer = Dense(2, activation='softmax', name='softmax')(x)
    
    
    net_final = Model(inputs=base_model.input, outputs=output_layer)
    
    
    net_final.compile(optimizer=Adam(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    # print(net_final.summary())
    
    
    return net_final
######################################################################################################
model = create_model_new()
EPOCHS = 20
histories = []
ls_val_acc = []
ls_test_acc = []
ls_test_f1 = []
ls_test_se = []
ls_test_sp = []

kfold = KFold(10, shuffle=False)
# kfold = KFold(5, shuffle=True, random_state=123)


import time


for f, (trn_ind, val_ind) in enumerate(kfold.split(img_df_over)):
    model = create_model_new()
    print(); print("#"*50)
    print("Fold: ",f+1)
    print("#"*50)
    train_ds = t_datagen.flow_from_dataframe(img_df_over.loc[trn_ind,:], 
                                       x_col='PATH', y_col='LABELS',
                                       target_size=(width,height),
                                       class_mode = 'categorical', color_mode = 'rgb',
                                       batch_size = 16, shuffle = False)
    val_ds = v_datagen.flow_from_dataframe(img_df_over.loc[val_ind,:], 
                                     x_col='PATH', y_col='LABELS',
                                     target_size=(width,height),
                                     class_mode = 'categorical', color_mode = 'rgb',
                                     batch_size = 16, shuffle = False)
    for cls, idx in train_ds.class_indices.items():
      print('Class #{} = {}'.format(idx, cls))
    # Define start and end epoch for each folds
    fold_start_epoch = f * EPOCHS
    fold_end_epoch = EPOCHS * (f+1)
    

    
    # fit
    history=model.fit(train_ds, initial_epoch=fold_start_epoch , epochs=fold_end_epoch, 
                  validation_data=val_ds, shuffle=False)

    # store history for each folds
    histories.append(history)
    
    print("#"*50)
    val_loss, val_acc = model.evaluate(val_ds)
    print(f'Model accuracy on val: {val_acc*100:6.2f}')
    print("#"*50)
    
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Model accuracy on test: {test_acc*100:6.2f}')
    print("#"*50)
    
    
    
      
    
    vY_pred = model.predict_generator(val_ds)
    vy_pred = np.argmax(vY_pred, axis=1)
    print('VAL Matrix')
    print(confusion_matrix(val_ds.classes, vy_pred))
    print('Classification Report')
    target_names = ['bad', 'good']
    print(classification_report(val_ds.classes, vy_pred, target_names=target_names))
    start = time.process_time()
    Y_pred = model.predict_generator(test_ds)
    end = time.process_time()
    y_pred = np.argmax(Y_pred, axis=1)
    print('TEST Matrix')
    print(confusion_matrix(test_ds.classes, y_pred))
    print('Classification Report')
    target_names = ['bad', 'good']
    print(classification_report(test_ds.classes, y_pred, target_names=target_names))
    
    print("f1=  "+str(f1_score(test_ds.classes, y_pred,average='macro')))
    print("acc=  "+str(accuracy_score(test_ds.classes, y_pred)))
    print("pos_recall(sensitivity)=  "+str(recall_score(test_ds.classes, y_pred, pos_label=0)))
    print("neg_recall(specificity)=  "+str(recall_score(test_ds.classes, y_pred, pos_label=1)))
    print("#"*50)
    print("執行時間：%f 秒" % (end - start))
    if EPOCHS>=15:
        ls_val_acc.append(val_acc)
        ls_test_acc.append(test_acc)
        ls_test_f1.append(f1_score(test_ds.classes, y_pred,average='macro'))
        ls_test_se.append(recall_score(test_ds.classes, y_pred, pos_label=0))
        ls_test_sp.append(recall_score(test_ds.classes, y_pred, pos_label=1))
    
        tes=str(int(accuracy_score(test_ds.classes, y_pred)*100))
        
        model.save(str(data_path)+'/'+str(f)+'_'+tes+'vgg.h5')
        
        test_dataset=test_dataset.assign(preds=y_pred)
        test_dataset.to_csv(str(data_path)+'/'+str(f)+'_'+tes+'test_dataset.csv', sep=',', encoding='utf-8',header=True,index=False)
    
    if EPOCHS==20:
        try:
            xlog = pd.read_csv(str(data_path)+'/'+na+'xlog.csv')
            newxlog = pd.DataFrame()
            newxlog=newxlog.assign(ls_test_acc=ls_test_acc)
            newxlog=newxlog.assign(ls_test_f1=ls_test_f1)
            newxlog=newxlog.assign(ls_test_se=ls_test_se)
            newxlog=newxlog.assign(ls_test_sp=ls_test_sp)
            newxlog = pd.concat([xlog,newxlog],axis=0,ignore_index=True)
            newxlog.to_csv(str(data_path)+'/'+na+'xlog.csv', sep=',', encoding='utf-8',header=True,index=False)
        except:
            xlog = pd.DataFrame()
            xlog=xlog.assign(ls_test_acc=ls_test_acc)
            xlog=xlog.assign(ls_test_f1=ls_test_f1)
            xlog=xlog.assign(ls_test_se=ls_test_se)
            xlog=xlog.assign(ls_test_sp=ls_test_sp)
    
            xlog.to_csv(str(data_path)+'/'+na+'xlog.csv', sep=',', encoding='utf-8',header=True,index=False)
        ls_val_acc = []
        ls_test_acc = []
        ls_test_f1 = []
        ls_test_se = []
        ls_test_sp = []
    
    # model.save("VGG16_a.h5")
    
print(str(data_path)+'  e='+str(EPOCHS)+'_')

print("#"*50)
print('VAL_avg='+str(sum(ls_val_acc) / len(ls_val_acc)))
for i in range(len(ls_val_acc)):
    print(ls_val_acc[i])
print("#"*50)
print('TEST_avg='+str(sum(ls_test_acc) / len(ls_test_acc)))
TEST_avg=sum(ls_test_acc) / len(ls_test_acc);TEST_avg=str(int(round(TEST_avg,2)*100))
for i in range(len(ls_test_acc)):
    print(ls_test_acc[i])
print("#"*50)
print('TEST_F1='+str(sum(ls_test_f1) / len(ls_test_f1)))
print('TEST_SE='+str(sum(ls_test_se) / len(ls_test_se)))
print('TEST_SP='+str(sum(ls_test_sp) / len(ls_test_sp)))


# test_gen = ImageDataGenerator(rescale=1/255.0) # just rescaling for test data
# test_ds = test_gen.flow_from_dataframe(test_dataset, x_col='PATH', y_col='LABELS',
#                                        target_size=(width,height),
#                                        class_mode = 'categorical', 
#                                        color_mode = 'rgb',
#                                        batch_size = 16)
  
# test_loss, test_acc = model.evaluate(test_ds)
# print(f'Model accuracy on test: {test_acc*100:6.2f}')


def plot_acc_loss(histories):
  acc, val_acc = [], []
  loss, val_loss = [], []
  for i in range(len(histories)):
    acc += histories[i].history['accuracy']
    val_acc += histories[i].history['val_accuracy']

    loss += histories[i].history['loss']
    val_loss += histories[i].history['val_loss']
  
  total_epoch = len(histories) * len(history.epoch) # num of fold * each epoch 
  epochs_range = range(total_epoch)

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()
  
# plot accuracy and loss of train and validation dataset
plot_acc_loss(histories)



###################################################


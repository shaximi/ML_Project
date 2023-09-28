import cv2
import numpy as np
import matplotlib.pyplot as plt
'''SHAP Image Explain heatmap counting'''


first=True
for i in range(32):
    img1 = cv2.imread("D:/shen/d/2class/k-ford_test/256a_v1_final/tr/shap_map/r/"+str(i)+".png")
    img1 = cv2.resize(img1,(256,256))
    if first==True:#初始清空
        fmap= np.empty((256,256))
        for i in range(0,256):#列
            for j in range(0,256):#欄
                fmap[j,i]=0
    
    (B,G,R) = cv2.split(img1)
    
    for i in range(256):
        for j in range(256):
            if G[i][j] <= 180 and R[i][j] >= 220 and B[i][j] <= 200:#紅色
                print(i,j,"RED")
                tp=fmap[i][j]
                tp+=1
                fmap[i][j]=tp
    first=False

cv2.imshow("img1",fmap)
cv2.waitKey(0)
cv2.destroyAllWindows()


####################設定門檻合成特徵熱力圖######################
for x in range(10):#門檻值由1跑到10
    fulll= np.empty((256,256))
    for i in range(0,256):#列
        for j in range(0,256):#欄
            fulll[j,i]=fmap[j,i] #複製到fulll做處理
    for i in range(0,256):#列
        for j in range(0,256):#欄
            fu=0;ma=0;
            del fu;del ma
            fu=fulll[j,i]
            if fu>x:
                fulll[j,i]=255
            else:
                fulll[j,i]=0
                
    plt.imshow(fulll)
    plt.show()
    plt.imsave(str(x)+"shap_map.png",fulll)

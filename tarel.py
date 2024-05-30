import cv2
import numpy as np


def ColorCorrect(orign_img,img): #色彩调整
    # img = np.float64(img) / 255
    # orign_img=np.float64(orign_img)/255
    B,G,R=np.zeros_like(img[:,:,0]),np.zeros_like(img[:,:,0]),np.zeros_like(img[:,:,0])
    for i in range(3):
        orign_log=np.log(orign_img[:,:,i])
        later_log=np.log(img[:,:,i])
        pre_mse=np.std(orign_log)
        pre_mean=np.mean(orign_log)
        later_mse=np.std(later_log)
        later_mean=np.mean(later_log)
        if i ==0:
             B=np.power(img[:,:,i],later_mean/later_mse*np.exp(pre_mean-pre_mse*later_mean/later_mse))
        elif i==1:
             G=np.power(img[:,:,i],later_mean/later_mse*np.exp(pre_mean-pre_mse*later_mean/later_mse))
        else:
             R=np.power(img[:,:,i],later_mean/later_mse*np.exp(pre_mean-pre_mse*later_mean/later_mse))
    U=cv2.merge([B,G,R])
    U=(U*255).astype(np.uint8)
    G=cv2.cvtColor(U, cv2.COLOR_BGR2GRAY)
    MG=np.max(G)
    result=np.zeros_like(U)
    for i in range(3):
         result[:,:,i]=U[:,:,i]/(1+(1/255-1/MG)*G)
    return result

def local_smooth(img,u):  #动态局部滤波
    hImg = img.shape[0]
    wImg = img.shape[1]
    smax=19
    m, n = smax, smax
    hPad = int((m-1) / 2)
    wPad = int((n-1) / 2)
    imgPad = np.pad(img.copy(), ((hPad, m-hPad-1), (wPad, n-wPad-1)), mode="edge")
    result = np.zeros(img.shape)
    for i in range(hPad, hPad+hImg):
        for j in range(wPad, wPad+wImg):
            ksize =min(u[i-hPad][j-wPad],smax)
            k = int(ksize/2)
            pad = imgPad[i-k:i+k+1, j-k:j+k+1]  # 邻域 Sxy, m*n
            result[i-hPad, j-wPad] = np.median(pad)
    return result

def white_balance(img):
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r
 
    return img

def tarel(img,local=False,color_cor=False):
    s_v=5
    p=0.95
    balance_img=white_balance(img)
    W=np.min(balance_img,axis=2)
    A=cv2.medianBlur(W,s_v)
    B=W-A
    B=A-cv2.medianBlur(np.uint8(B),s_v)
    min_t=np.minimum(np.uint8(p*B),np.uint8(W))
    V = np.maximum(min_t,0)
    V = cv2.blur(V,(5,5))
    V = np.float32(V) / 255

    R_dehazy = np.zeros((V.shape[0],V.shape[1],3), dtype=np.float32)
    # R=
    original_wb = np.float32(balance_img) / 255

    for i in range(3):
        R_dehazy[:,:,i] = (original_wb[:,:,i] - V) / (1 - V)
    
    R_dehazy = R_dehazy /255
    R_dehazy = np.clip(R_dehazy,0,1)
    R_dehazy= cv2.normalize(R_dehazy,None,0,255,cv2.NORM_MINMAX)
    # color_cor=True
    if local:
         u=1/(1-V)
         B,G,R=cv2.split(R_dehazy)
         B,G,R=local_smooth(B,u),local_smooth(G,u),local_smooth(R,u)
         R_dehazy=cv2.merge([B,G,R])
    if color_cor:
        R_dehazy=ColorCorrect(original_wb,R_dehazy)
    result=R_dehazy.astype(np.uint8)
    result+=40
    return result

if __name__ == "__main__":
    img=cv2.imread('collect/hazy/4.png',flags=cv2.IMREAD_UNCHANGED)
    m = tarel(img)
    cv2.imshow("orign",img)
    cv2.imshow("result",m)
    cv2.imwrite('collect/tarel_result/4.png',m)
    key=cv2.waitKey(0)

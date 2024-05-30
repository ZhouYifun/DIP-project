import darkChannel
import CLAHE
import os
import tarel
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.color import deltaE_ciede2000 as CIEDE
import time

def clahe_test(filepath_GT,filepath_hazy,file_result):
    filename_hazy=os.listdir(filepath_hazy)
    filename_GT=os.listdir(filepath_GT)
    psnr_gt_hazy=[]
    psnr_gt_result=[]
    mse_gt_hazy=[]
    mse_gt_result=[]
    ssim_gt_hazy=[]
    ssim_gt_result=[]
    ciede_gt_hazy=[]
    ciede_gt_result=[]
    start_time=time.time()
    for i in range(len(filename_GT)):
        hazy_name=filename_hazy[i]
        GT_name=filename_GT[i]
        hazy_path=os.path.join(filepath_hazy,hazy_name)
        GT_path=os.path.join(filepath_GT,GT_name)
        hazy_img=cv2.imread(hazy_path,1)
        GT_img=cv2.imread(GT_path,1)
        hazy_img=cv2.resize(hazy_img,None,fx=0.2,fy=0.2)
        GT_img=cv2.resize(GT_img,None,fx=0.2,fy=0.2)
        h,w=hazy_img.shape[:2]
        result=np.copy(hazy_img)
        # result[:h//2,:w//2]= CLAHE.CLAHE(hazy_img[:h//2,:w//2])
        # result[h//2:,:w//2]=CLAHE.CLAHE(hazy_img[h//2:,:w//2])
        # result[:h//2,w//2:]=CLAHE.CLAHE(hazy_img[:h//2,w//2:])
        # result[h//2:,w//2:]=CLAHE.CLAHE(hazy_img[h//2:,w//2:])
       
        result=CLAHE.CLAHE(result)
        mse_gt_hazy.append(MSE(GT_img,hazy_img))
        mse_gt_result.append(MSE(GT_img,result))
        ssim_gt_hazy.append(SSIM(GT_img,hazy_img, multichannel=True,channel_axis=-1))
        ssim_gt_result.append(SSIM(GT_img,result,multichannel=True,channel_axis=-1))
        psnr_gt_hazy.append(PSNR(GT_img,hazy_img))
        psnr_gt_result.append(PSNR(GT_img,result))
        ciede_gt_hazy.append(CIEDE(GT_img,hazy_img).mean())
        ciede_gt_result.append(CIEDE(GT_img,result).mean())
        print(i)
        
        result_name="%d.png"%(i)
        result_path=os.path.join(file_result,result_name)
        cv2.imwrite(result_path, result)

    end_time=time.time()
    print(len(mse_gt_hazy))
    print("CLAHE算法结果：")
    print("运行时间：'%f'"%(end_time-start_time))
    print("MSE:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(mse_gt_hazy),np.mean(mse_gt_result)))
    print("PSNR:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(psnr_gt_hazy),np.mean(psnr_gt_result)))
    print("SSIM:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ssim_gt_hazy),np.mean(ssim_gt_result)))
    print("CIEDE2000:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ciede_gt_hazy),np.mean(ciede_gt_result)))


def dark_channel_test(filepath_GT,filepath_hazy,file_result):
    filename_hazy=os.listdir(filepath_hazy)
    filename_GT=os.listdir(filepath_GT)
    psnr_gt_hazy=[]
    psnr_gt_result=[]
    mse_gt_hazy=[]
    mse_gt_result=[]
    ssim_gt_hazy=[]
    ssim_gt_result=[]
    ciede_gt_hazy=[]
    ciede_gt_result=[]
    start_time=time.time()
    for i in range(len(filename_GT)):
        hazy_name=filename_hazy[i]
        GT_name=filename_GT[i]
        hazy_path=os.path.join(filepath_hazy,hazy_name)
        GT_path=os.path.join(filepath_GT,GT_name)
        hazy_img=cv2.imread(hazy_path,1)
        GT_img=cv2.imread(GT_path,1)
        hazy_img=cv2.resize(hazy_img,None,fx=0.2,fy=0.2)
        GT_img=cv2.resize(GT_img,None,fx=0.2,fy=0.2)
        result=darkChannel.dark_channel(hazy_img)
        mse_gt_hazy.append(MSE(GT_img,hazy_img))
        mse_gt_result.append(MSE(GT_img,result))
        ssim_gt_hazy.append(SSIM(GT_img,hazy_img, multichannel=True,channel_axis=-1))
        ssim_gt_result.append(SSIM(GT_img,result,multichannel=True,channel_axis=-1))
        psnr_gt_hazy.append(PSNR(GT_img,hazy_img))
        psnr_gt_result.append(PSNR(GT_img,result))
        ciede_gt_hazy.append(CIEDE(GT_img,hazy_img).mean())
        ciede_gt_result.append(CIEDE(GT_img,result).mean())
        print(i)
        result_name="%d.png"%(i)
        result_path=os.path.join(file_result,result_name)
        cv2.imwrite(result_path, result)
    end_time=time.time()
    print(len(mse_gt_hazy))
    print("dark_channel算法结果：")
    print("运行时间：'%f'"%(end_time-start_time))
    print("MSE:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(mse_gt_hazy),np.mean(mse_gt_result)))
    print("PSNR:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(psnr_gt_hazy),np.mean(psnr_gt_result)))
    print("SSIM:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ssim_gt_hazy),np.mean(ssim_gt_result)))
    print("CIEDE2000:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ciede_gt_hazy),np.mean(ciede_gt_result)))

def tarel_test(filepath_GT,filepath_hazy,file_result):
    filename_hazy=os.listdir(filepath_hazy)
    filename_GT=os.listdir(filepath_GT)
    psnr_gt_hazy=[]
    psnr_gt_result=[]
    mse_gt_hazy=[]
    mse_gt_result=[]
    ssim_gt_hazy=[]
    ssim_gt_result=[]
    ciede_gt_hazy=[]
    ciede_gt_result=[]
    start_time=time.time()
    for i in range(len(filename_GT)):
        hazy_name=filename_hazy[i]
        GT_name=filename_GT[i]
        hazy_path=os.path.join(filepath_hazy,hazy_name)
        GT_path=os.path.join(filepath_GT,GT_name)
        hazy_img=cv2.imread(hazy_path,1)
        GT_img=cv2.imread(GT_path,1)
        hazy_img=cv2.resize(hazy_img,None,fx=0.2,fy=0.2)
        GT_img=cv2.resize(GT_img,None,fx=0.2,fy=0.2)

        result=tarel.tarel(hazy_img)

        mse_gt_hazy.append(MSE(GT_img,hazy_img))
        mse_gt_result.append(MSE(GT_img,result))
        ssim_gt_hazy.append(SSIM(GT_img,hazy_img, multichannel=True,channel_axis=-1))
        ssim_gt_result.append(SSIM(GT_img,result,multichannel=True,channel_axis=-1,data_range=255))
        psnr_gt_hazy.append(PSNR(GT_img,hazy_img))
        psnr_gt_result.append(PSNR(GT_img,result))
        ciede_gt_hazy.append(CIEDE(GT_img,hazy_img).mean())
        ciede_gt_result.append(CIEDE(GT_img,result).mean())
 
        print(i)
        result_name="%d.png"%(i)
        result_path=os.path.join(file_result,result_name)
        cv2.imwrite(result_path, result)
    end_time=time.time()
    print(len(mse_gt_hazy))
    print("tarel算法结果：")
    print("运行时间：'%f'"%(end_time-start_time))
    print("MSE:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(mse_gt_hazy),np.mean(mse_gt_result)))
    print("PSNR:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(psnr_gt_hazy),np.mean(psnr_gt_result)))
    print("SSIM:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ssim_gt_hazy),np.mean(ssim_gt_result)))
    print("CIEDE2000:")
    print("GT_hazy:'%f'  GT_result:'%f'"%(np.mean(ciede_gt_hazy),np.mean(ciede_gt_result)))

    
if __name__ == '__main__':
    filepath_GT="SOTS/gt"
    filepath_hazy="SOTS/hazy"
    file_result="SOTS/clahe"
    clahe_test(filepath_GT,filepath_hazy,file_result)
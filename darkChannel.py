import cv2
import numpy as np

def zmMinFilterGray(src, r=9):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

def get_dark_image(img, r=9):
    min_channel_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    dark_img = cv2.erode(min_channel_img, kernel)

    return dark_img

def guidedfilter(I, p, r, eps):  #导向滤波
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def atmospheric_light(img,eps):
    dark_img=get_dark_image(img,r=9)
    num=int(eps*(dark_img.shape[0]*dark_img.shape[1]))
    index=np.argsort(dark_img.ravel())[-num:]
    index=np.unravel_index(index,dark_img.shape)
    A=np.mean(img[index],axis=0)
    return A

def get_t(img,A,w,t0):
    t=1-w*get_dark_image(img/A,r=9)
    t=np.maximum(t,t0)
    return t

def dark_channel(img,eps=0.001, w=0.95, t0=0.1):
    img=img/255.0
    A=atmospheric_light(img,eps)
    t=get_t(img,A,w,t0)
    V1 = np.min(img, 2)                           # 得到暗通道图像
    # Dark_Channel = zmMinFilterGray(V1, 9)
    t=guidedfilter(V1,t,81,eps)
    J = (img - A) / t.reshape(t.shape[0], t.shape[1], 1) + A
    J = np.clip(J, 0, 1)

    return (J * 255).astype(np.uint8)


if __name__ == '__main__':
    img=cv2.imread('collect/hazy/4.png',flags=cv2.IMREAD_UNCHANGED)
    m = dark_channel(img)
    cv2.imshow("orign",img)
    cv2.imshow("result",m)
    cv2.imwrite('collect/dark_result/4.png',m)
    key=cv2.waitKey(0)


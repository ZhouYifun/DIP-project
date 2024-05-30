import cv2
import numpy as np
import math

def get_dark_image(img, r=9):
    min_channel_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    dark_img = cv2.erode(min_channel_img, kernel)

    return dark_img

def atmospheric_light(img,eps):
    dark_img=get_dark_image(img,r=9)
    num=int(eps*(dark_img.shape[0]*dark_img.shape[1]))
    index=np.argsort(dark_img.ravel())[-num:]
    index=np.unravel_index(index,dark_img.shape)
    A=np.mean(img[index],axis=0)
    return A

def sqr(x):
    return x * x

def norm(array):
    return math.sqrt(sqr(array[0]) + sqr(array[1]) + sqr(array[2]))

def avg(vals):
    return sum(vals) / len(vals)

def conv(xs, ys):
    ex = avg(xs)
    ey = avg(ys)
    return sum((xs-ex)*(ys-ey)) / len(xs)

def stress(input):
    data_max = np.max(input)
    data_min = np.min(input)
    temp = data_max - data_min

    output = (input - data_min) / temp
    output[output < 0.1] = 0.1
    return output

def getDehaze(scrimg, transmission, airlight):
    one = np.ones(transmission.shape, dtype=np.float32)
    B,G,R = cv2.split(scrimg)

    R = (R - (one - transmission) * airlight[2]) / transmission
    G = (G - (one - transmission) * airlight[1]) / transmission
    B = (B - (one - transmission) * airlight[0]) / transmission

    result = cv2.merge([B,G,R])
    return result

def getTransmission(input, airlight):
    normA = norm(airlight)

    # Calculate Ia
    dotresult = np.tensordot(input, airlight, axes=([2], [0])) / normA
    Ia = dotresult / normA

    # Calculate Ir
    input_norm = np.linalg.norm(input, axis=2)
    Ir = np.sqrt(sqr(input_norm) - sqr(Ia))

    # Calculate h
    h = (normA - Ia) / Ir

    # Estimate the eta
    Iapix = Ia.ravel()
    Irpix = Ir.ravel()
    hpix = h.ravel()

    eta = conv(Iapix, hpix) / conv(Irpix, hpix)

    # Calculate the transmission
    t = 1 - (Ia - eta * Ir) / normA
    trefined = stress(t)
    return trefined

def fattal(image):
    n_blocks=1
    scale = 1.0
    originRows, originCols = image.shape[:2]
    if scale < 1.0:
        resizedImage = cv2.resize(image, (int(originCols * scale), int(originRows * scale)))
    else:
        scale = 1.0
        resizedImage = image

    rows, cols = resizedImage.shape[:2]
    convertImage = resizedImage.astype(np.float32) / 255.0
    finalImage=np.zeros_like(convertImage)
    block_row=int(rows/n_blocks)
    block_col=int(cols/n_blocks)
    tmp_A = atmospheric_light(convertImage,eps=0.001)
    for i in range(n_blocks):
        for j in range(n_blocks):
            x_start=i*block_row
            x_end=(i+1)*block_row
            if i==n_blocks-1:
                x_end=rows
            y_start=j*block_col
            y_end=(j+1)*block_col
            if j==n_blocks-1:
                y_end=cols
            trans = getTransmission(convertImage[x_start:x_end,y_start:y_end], tmp_A)
            finalImage[x_start:x_end,y_start:y_end] = getDehaze(convertImage[x_start:x_end,y_start:y_end], trans, tmp_A)
    finalImage=finalImage+0.4
    
    cv2.normalize(finalImage,None,0,255,cv2.NORM_MINMAX)
    return finalImage


if __name__ == "__main__":
    loc = "3.png"

    image = cv2.imread(loc)
    cv2.imshow("hazyimage", image)
    result=fattal(image)
    cv2.imshow('result',result)
    cv2.waitKey(0)
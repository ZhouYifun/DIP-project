import numpy as np
import cv2

def CLAHE(img, n_blocks=8, threshold=50.0):
    B,G,R=cv2.split(img)
    B,G,R=apply_clache(B,n_blocks,threshold),apply_clache(G,n_blocks,threshold),apply_clache(R,n_blocks,threshold)
    new_img=cv2.merge([B,G,R])
    return new_img

def clip_histogram(hist, clip_limit):
    """ Clip the histogram and redistribute excess pixels."""
    excess = np.sum(hist[hist > clip_limit] - clip_limit)
    hist = np.clip(hist, 0, clip_limit)
    hist += excess // hist.size
    return hist

def calc_histogram_cdf(hist):
    """ Calculate the cumulative distribution function (CDF) of a histogram."""
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized

def contrast(img,threshold): #得到限制对比度后的直方图
    hist=np.histogram(img, bins=256, range=(0, 256))[0]
    clip_his=clip_histogram(hist,threshold)
    hist_constr=calc_histogram_cdf(clip_his)
    cdf_m = np.ma.masked_equal(hist_constr, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf

def interpolate(img, maps):
    h, w = img.shape
    n_blocks = len(maps)
    block_h = int(h / n_blocks)
    block_w = int(w / n_blocks)
    # Interpolate every pixel using four nearest mapping functions
    new_img = np.empty_like(img)
    for i in range(h):
        for j in range(w):
            origin_val = img[i][j]

            r = int(
                np.floor(i / block_h - 0.5)
            )  # The row index of the left-up mapping function
            c = int(
                np.floor(j / block_w - 0.5)
            )  # The col index of the left-up mapping function

            x1 = (
                (i + 0.5) - (r + 0.5) * block_h
            ) / block_h  # The x-axis distance to the left-up mapping center
            y1 = (
                (j + 0.5) - (c + 0.5) * block_w
            ) / block_w  # The y-axis distance to the left-up mapping center

            # Four corners use the nearest mapping directly
            if r == -1 and c == -1:
                new_img[i][j] = maps[0][0][origin_val]
            elif r == -1 and c >= n_blocks - 1:
                new_img[i][j] = maps[0][-1][origin_val]
            elif r >= n_blocks - 1 and c == -1:
                new_img[i][j] = maps[-1][0][origin_val]
            elif r >= n_blocks - 1 and c >= n_blocks - 1:
                new_img[i][j] = maps[-1][-1][origin_val]
            # Four border case using the nearest two mapping
            elif r == -1 or r >= n_blocks - 1:
                if r == -1:
                    r = 0
                else:
                    r = n_blocks - 1
                left = maps[r][c][origin_val]
                right = maps[r][c + 1][origin_val]
                new_img[i][j] = (1 - y1) * left + y1 * right
            elif c == -1 or c >= n_blocks - 1:
                if c == -1:
                    c = 0
                else:
                    c = n_blocks - 1
                up = maps[r][c][origin_val]
                bottom = maps[r + 1][c][origin_val]
                new_img[i][j] = (1 - x1) * up + x1 * bottom
            # Bilinear interpolate for inner pixels
            else:
                lu = maps[r][c][origin_val]  # Mapping value of the left up cdf
                lb = maps[r + 1][c][origin_val]
                ru = maps[r][c + 1][origin_val]
                rb = maps[r + 1][c + 1][origin_val]
                new_img[i][j] = (1 - y1) * ((1 - x1) * lu + x1 * lb) + y1 * \
                    ((1 - x1) * ru + x1 * rb)
    new_img = new_img.astype("uint8")
    return new_img

def apply_clache(img,n_blocks,threshold):
    H,W=img.shape
    maps=[]
    block_h=int(H/n_blocks)
    block_w=int(W/n_blocks)
    # for i in range(n_blocks):
    #     maps[i]=[]
    for i in range(n_blocks):
        row_map=[]
        for j in range(n_blocks):
            x_start=i*block_h
            x_end=(i+1)*block_h
            if i==n_blocks-1:
                x_end=H
            y_start=j*block_w
            y_end=(j+1)*block_w
            if j==n_blocks-1:
                y_end=W
            img_piece=img[x_start:x_end,y_start:y_end]
            contr_his=contrast(img_piece,threshold)
            row_map.append(contr_his)
        maps.append(row_map)
    
    new_img=interpolate(img,maps)
    return new_img


if __name__ == '__main__':
    img=cv2.imread('collect/hazy/4.png',flags=cv2.IMREAD_UNCHANGED)
    m = CLAHE(img)
    cv2.imshow("orign",img)
    cv2.imshow("result",m)
    cv2.imwrite('collect/clahe_result/4.png',m)
    key=cv2.waitKey(0)
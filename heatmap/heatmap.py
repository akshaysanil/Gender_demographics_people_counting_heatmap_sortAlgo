import os
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import time
from datetime import datetime 
import io


def _ploat_heatmap(x,y,s,height,width,bins=1000,intensity=3.0):
    heatmap,xedges,yedges = np.histogram2d(x ,y ,bins=bins, range=[[0,width],[0,height]])
    # heatmap = gaussian_filter(heatmap,sigma=s)
    heatmap = gaussian_filter(heatmap, sigma=s * intensity)
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    return heatmap.T, extent

def heatmap(img1,centroid,height, width):
    time_stamp = time.time()
    narr = np.array(centroid)
    print(narr)
    x,y = narr.T
    img, extent = _ploat_heatmap(x,y,32,height, width)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,extent=extent,cmap=cm.jet,aspect='auto')
    heatmap_path = os.path.join('hmp_100per_intencity/hmp.png')
    fig.savefig(heatmap_path)
    alpha = 0.8
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img.shape[:2] != img1.shape[:2]:
        heatmap_img = cv2.resize(heatmap_img, (width, height))   
    blended_img = cv2.addWeighted(heatmap_img,alpha, img1, 1-alpha, 0 )
    # formated_time = datetime.fromtimestamp(time_stamp).strftime('%Y_%m_%d__%H_%M_%S')

    # cv2.imwrite(f'hmp_50per_intencity/hmp_result_{formated_time}.png',blended_img)

    print(img.shape)
    return blended_img


# # def heatmap(img1, centroid, height, width):
# #     time_stamp = time.time()
# #     narr = np.array(centroid)
# #     print(narr)
# #     x, y = narr.T
# #     img, extent = _ploat_heatmap(x, y, 32, height, width)
    
# #     fig = plt.figure(frameon=False)
# #     ax = plt.Axes(fig, [0., 0., 1., 1.])
# #     ax.set_axis_off()
# #     fig.add_axes(ax)
# #     ax.imshow(img, extent=extent, cmap=cm.jet, aspect='auto')
    
# #     # Convert the figure to an image without saving
# #     buf = io.BytesIO()
# #     plt.savefig(buf, format='png')
# #     buf.seek(0)
# #     heatmap_img = plt.imread(buf)
# #     plt.close(fig)
    
# #     alpha = 0.8
# #     if heatmap_img.shape[:2] != img1.shape[:2]:
# #         heatmap_img = cv2.resize(heatmap_img, (width, height))   
# #     blended_img = cv2.addWeighted(heatmap_img, alpha, img1, 1 - alpha, 0)
    
# #     return blended_img

# def heatmap(img1, centroid, height, width):
#     time_stamp = time.time()
#     narr = np.array(centroid)
#     x, y = narr.T
#     img, extent = _ploat_heatmap(x, y, 32, height, width)
    
#     fig = plt.figure(frameon=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(img, extent=extent, cmap=cm.jet, aspect='auto')
    
#     # Convert the figure to an image without saving
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     heatmap_img = plt.imread(buf)
#     plt.close(fig)
    
#     alpha = 0.8
    
#     # Resize heatmap_img to match the size of img1
#     heatmap_img_resized = cv2.resize(heatmap_img, (img1.shape[1], img1.shape[0]))
#     print(heatmap_img_resized.shape)
#     print(img1.shape)
    
#     blended_img = cv2.addWeighted(heatmap_img_resized, alpha, img1, 1 - alpha, 0)
    
#     return blended_img
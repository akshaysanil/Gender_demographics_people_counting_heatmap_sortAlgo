import os
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import time


def _plot_heatmap(x, y, s, height, width, bins=1000, intensity=3.0):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, width], [0, height]])
    heatmap = gaussian_filter(heatmap, sigma=s * intensity)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def heatmap_with_trails(img1, centroid, data_deque, height, width):

    time_stamp = time.time()
    narr = np.array(centroid)
    x, y = narr.T
    img, extent = _plot_heatmap(x, y, 32, height, width)
 
    # for id, trail in data_deque.items():
    #     print('trail : ',trail)
    #     for i in range(1,len(trail)):
    #         # print('trail indivudual : ',i)
    #         if trail[i-1] is None or trail[i] is None:
    #             # print('inside-------------')
    #             continue
    #         thickness = 6

    #         cv2.line(img1,trail[i-1],trail[i],[255,255,0],thickness)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, extent=extent, cmap=cm.jet, aspect='auto')

    # creating_heatmap dir
    heatmap_dir = 'hmp_100per_intencity'
    os.makedirs(heatmap_dir, exist_ok=True)


    heatmap_path = os.path.join(heatmap_dir,'hmp_{}.png'.format(time_stamp))
    fig.savefig(heatmap_path)
    
    alpha = 0.8
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img.shape[:2] != img1.shape[:2]:
        heatmap_img = cv2.resize(heatmap_img, (width, height))
    blended_img = cv2.addWeighted(heatmap_img, alpha, img1, 1 - alpha, 0)

    for id, trail in data_deque.items():
        print('trail : ',trail)
        for i in range(1,len(trail)):
            if trail[i-1] is None or trail[i] is None:
                continue
            thickness = 6
            cv2.line(blended_img,trail[i-1],trail[i],[255,255,0],thickness)

    overlayed_img_path = os.path.join("./heatmap_on_img.jpg")
    print("saving overlayed heatmap: {}".format(overlayed_img_path))
    return blended_img
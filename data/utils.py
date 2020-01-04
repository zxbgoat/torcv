import random
import numpy as np


def drawbbox(img, anns, tk=2, color=None):
    if str(img.dtype).startswith('float'):
        rd, up = random.uniform, 1.0
    else:
        rd, up = random.randint, 255
    for i in range(len(anns)):
        color = np.array([rd(0, up), rd(0, up), rd(0, up)])
        x, y = anns[i][0], anns[i][1]
        xl, xr = round(x-tk/2), round(x+tk/2)
        yt, yb = round(y-tk/2), round(y+tk/2)
        x, y, w, h = list(map(round, anns[i][:4]))
        img[yt:yb, x:x+w, :] = color
        img[y:y+h, xl:xr, :] = color
        img[yt+h:yb+h, x:x+w, :] = color
        img[y:y+h, xl+w:xr+w, :] = color
    return img

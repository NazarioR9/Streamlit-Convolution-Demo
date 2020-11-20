import os
import numpy as np
from PIL import Image


def get_edge_filter():
    return np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])


def get_identity_filter():
    return np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])

def read_img(path):
    return Image.open(path)
    
def resize(img, size=(64, 64)):
    return img.resize(size)


def grayscale(img, mode='L'):
    return img.convert(mode)
    
def read_for_conv(path, size=None):
    img = read_img(path)
    if size:
    	img = resize(img, size)
    return grayscale(img) 

def pad(arr, size):
    return np.pad(arr, size//2)


def save_img(img, fn):
    img.save(fn)
    
def scale(img, eps=1e-6):
    mean = img.mean()
    std = img.std()
    
    img_norm = (img - mean) / (std + eps)
    _min, _max = img_norm.min(), img_norm.max()
    
    img_norm = (img_norm - _min) / (_max - _min)
    
    return img_norm


def my_convolution(img_src, kernel=get_edge_filter()):
    img_src = np.array(img_src, dtype=np.float)
    new_img = np.zeros_like(img_src)
    pad_img = pad(img_src, kernel.shape[1])

    hs, ws = new_img.shape
    kh, kw = kernel.shape

    for w in range(ws):
        for h in range(hs):
            patch = pad_img[h: h + kh, w: w + kw]
            new_img[h, w] += np.sum(patch * kernel)

    return scale(new_img)


def main():
    path = './resized_image/'
    save_path = './convolved_images/'
    imgs = os.listdir(path)

    for fn in imgs:
        img = read_img(path + fn)
        kernel = get_edge_filter()
        # kernel = get_identity_filter()
        img = my_convolution(img, kernel)

        img = Image.fromarray(img).convert('L')
        save_img(img, save_path + fn)


if __name__ == "__main__":
    main()

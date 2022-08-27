import numpy as np
from segmentation import *
from coding import *
import os
import tkinter as tk

def compare_codes(a, b, mask_a, mask_b, rotation=False):
    """Compares two codes and calculates Jaccard index.

    a: Code of the first iris
    b: Code of the second iris
    mask_a: Mask of the first iris
    mask_b: Mask of the second iris
    rotation: Maximum cyclic rotation of the code. If this argument is greater than zero, the function will
        return minimal distance of all code rotations. If this argument is False, no rotations are calculated.

    return: Distance between two codes.
    """
    if rotation:
        d = []
        for i in range(-rotation, rotation + 1):
            c = np.roll(b, i, axis=1)
            mask_c = np.roll(mask_b, i, axis=1)
            d.append(np.sum(np.remainder(a + c, 2) * mask_a * mask_c) / np.sum(mask_a * mask_c))
        return np.min(d)
    return np.sum(np.remainder(a + b, 2) * mask_a * mask_b) / np.sum(mask_a * mask_b)


def encode_photo(image):
    """
    Finds the pupil and iris of the eye, and then encodes the unravelled iris.

    image: Image of an eye
    return: Encoded iris (code, mask)
    """
    img = preprocess(image)
    x, y, r = find_pupil(img)
    x_iris, y_iris, r_iris = find_iris(img, x, y, r)
    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    return iris_encode(iris)


def save_codes(data):
    """
    Takes data, and saves encoded images to 'codes' directory.
    """
    for i in range(len(data['data'])):
        print("{}/{}".format(i, len(data['data'])))
        image = cv2.imread(data['data'][i])
        try:
            code, mask = encode_photo(image)
            np.save('codes\\code{}'.format(i), np.array(code))
            np.save('codes\\mask{}'.format(i), np.array(mask))
            np.save('codes\\target{}'.format(i), data['target'][i])
        except:
            np.save('codes\\code{}'.format(i), np.zeros(1))
            np.save('codes\\mask{}'.format(i), np.zeros(1))
            np.save('codes\\target{}'.format(i), data['target'][i])


def load_codes():
    """
    Loads codes saved by save_codes function.

    return: Codes, masks, and targets of saved images
    """
    codes = []
    masks = []
    targets = []
    i = 0
    while os.path.isfile('codes\\code{}.npy'.format(i)):
        code = np.load('codes\\code{}.npy'.format(i))
        if code.shape[0] != 1:
            codes.append(code)
            masks.append(np.load('codes\\mask{}.npy'.format(i)))
            targets.append(np.load('codes\\target{}.npy'.format(i)))
        i += 1
    return np.array(codes), np.array(masks), np.array(targets)

if __name__ == '__main__':
    data = load()['data']
    image = cv2.imread(data[1])
    i1=cv2.resize(image,(400,400))
    image2 = cv2.imread(data[1])
    i2=cv2.resize(image2,(400,400))
    print(image.shape)
    cv2.imshow('test1', i1)
    print(image2.shape)
    cv2.imshow('test2', i2)
    code, mask = encode_photo(image)
    code2, mask2 = encode_photo(image2)
    result=compare_codes(code, code2, mask, mask2)
    root = tk.Tk()
    T = tk.Text(root, height=4, width=40)
    T.pack()
    if result<0.2 :
        T.insert(tk.END, "Match Successful\nWelcome!!!\n")
        tk.mainloop()
    else :
         T.insert(tk.END, "Match Unsuccessful\nTry Again!!!\n")
         tk.mainloop()

import cv2
import numpy as np


def gau_noise(image):
    image = image.astype(np.float32)
    temp = image.copy()
    cv2.randu(temp, (-50), (50))
    image += temp
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = rotation(image)
    return image, (image.shape[:2])


def rotation(image):
    image = image.astype(np.float32)
    rows, cols = image.shape[:2]
    angle = np.random.randint(0, 180)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    res = cv2.warpAffine(image, M, (cols, rows)).astype(np.uint8)
    return res


def illumination(image):
    image = image.astype(np.int32)
    image += np.random.randint(-50, 50)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = rotation(image)
    return image, (image.shape[:2])


def scaling(image):
    height, width = image.shape[:2]
    scaling = np.random.uniform(0.4, 1.2)
    res = cv2.resize(image, (int(scaling * width), int(scaling * height)), interpolation=cv2.INTER_CUBIC)
    res = rotation(res)
    return res, (int(scaling * width), int(scaling * height))


def augmenting(image):
    n = np.random.randint(3)
    if n == 0:
        return scaling(image)
    elif n == 1:
        return illumination(image)
    else:
        return gau_noise(image)

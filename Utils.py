from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from os import path
import os
import tensorflow as tf
import codecs
import Csts
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure

HAND_WRITTEN_CHAR_FILE_NAME = 'images/written_char.png'

def resize_image(filename):

    image_grey = Image.open(filename).convert('L')
    resized_image = image_grey.resize(( Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), Image.ANTIALIAS)
    image_np_array = np.asarray(resized_image)
    image_np_array = image_np_array.reshape([Csts.IMAGE_SIZE, Csts.IMAGE_SIZE, 1])
    return image_np_array

def prepare_image(filename):

    image_grey = Image.open(filename).convert('L')
    resized_image = image_grey.resize(( Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), Image.ANTIALIAS)
    image_np_array = np.asarray(resized_image) / 255.0
    return image_np_array.reshape([-1, Csts.IMAGE_SIZE, Csts.IMAGE_SIZE, 1])

def prepare_image2(filename):

    image_grey = Image.open(filename).convert('L')
    resized_image = image_grey.resize(( Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), Image.ANTIALIAS)
    image_np_array = np.asarray(resized_image)
    image_np_array = image_np_array.reshape([-1, Csts.IMAGE_SIZE, Csts.IMAGE_SIZE, 1])
    image_np_array = preprocess_image(image_np_array)
    return image_np_array

# image is a png image
# returns a png image
def crop_image(image, tolerance=255):

    print("In crop_image")
    data = np.array(image)
    gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    print("shape2:", gray_image.shape)
    mask = gray_image < tolerance
    index = np.ix_(mask.any(1), mask.any(0))
    cropped_image_data = gray_image[index]
    return Image.fromarray(cropped_image_data)

def increase_contrast2(img):
    img = exposure.equalize_hist(img)
    return img

# Center image pixels values in [-0.5, +0.5]
def center_image_mean(img):
    img = img -0.5
    return img

def preprocess_image(img):
    contrasted_image = increase_contrast2(img)
    centered_image = center_image_mean(contrasted_image)
    centered_image = centered_image+0.01
    return centered_image

def increase_contrast(npImage):
    # Open the input image as numpy array, convert to greyscale and drop alpha
    # npImage = np.array(npImage.convert("L"))
    npImage = npImage.reshape(64, 64)
    # Get brightness range - i.e. darkest and lightest pixels
    min = np.min(npImage)  # result=144
    max = np.max(npImage)  # result=216

    # Make a LUT (Look-Up Table) to translate image values
    LUT = np.zeros(256, dtype=np.uint8)
    LUT[min:max + 1] = np.linspace(start=0, stop=255, num=(max - min) + 1, endpoint=True, dtype=np.uint8)

    # Apply LUT and save resulting image
    return Image.fromarray(LUT[npImage])

def get_size(image_path):

    if (not path.exists(image_path)):
        raise Exception('Image path {} does not exist !'.format(image_path))
    file_number = 0
    for base, dirs, files in os.walk(image_path):
        for file in files:
            file_number += 1
    return file_number

def get_class_size(image_path):

    if (not path.exists(image_path)):
        raise Exception('Image path {} does not exist !'.format(image_path))
    dir_number = 0
    for base, dirs, files in os.walk(image_path):
        for directories in dirs:
            dir_number += 1
    return dir_number


def convert_png_to_jpg(image_file_name) :
    im = Image.open(image_file_name + "png")
    rgb_im = im.convert('RGB')
    rgb_im.save(image_file_name+'.jpg')


# image_file_name is a png image
def set_image_background_to_white(image):
    data = np.array(image)
    alpha1 = 0  # Original value
    r2, g2, b2, alpha2 = 255, 255, 255, 255  # Value that we want to replace it with
    red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (alpha == alpha1)
    data[:, :, :4][mask] = [r2, g2, b2, alpha2]
    img = Image.fromarray(data)
    return img


def r(start_time):

    delta = datetime.now() - start_time
    ms = delta.seconds * 1000 + delta.microseconds / 1000
    if (ms >1000):
        return str(round( ms/1000, 2)) + " s"
    else:
        return str(round(ms, 2)) + " ms"

def get_label_char_dico(file):

    path = os.getcwd() + os.sep + file
    char_label_dictionary = load_char_label_dico(path)
    label_char_dico = get_label_char_map(char_label_dictionary)
    return label_char_dico

def augment(self, images):

    if self.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    if self.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    elif self.random_contrast:
        images = tf.image.random_contrast(images, 0.9, 1.1)  #  0.8 1.2
    return images

def get_label_char_map(character_label_dico):
    inverted_map = {v: k for k, v in character_label_dico.items()}
    return inverted_map

@property
def size(self):
    return len(self.labels)

def get_next_element(self):
    next_element = self.iterator.get_next()
    return next_element

def load_char_label_dico(filePath):

    print("Loading CharLabelDico ... ")
    start_time = datetime.now()
    charLabelMap = {}
    with codecs.open(filePath, 'r', 'gb2312') as f:
        for line in f:
            lineWithoutCR = line.split("\n")[0]
            splitted = lineWithoutCR.split(" ")
            char = splitted[0]
            label = int(splitted[1])
            charLabelMap[char] = label
    print("Execution time: %s s." % r(start_time))
    return charLabelMap


if __name__ == '__main__':
    filename = 'image/xia.png'
    image_grey = Image.open(filename).convert('L')
    image = crop_image(image_grey)


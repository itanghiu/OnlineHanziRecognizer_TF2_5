from Data import Data
import os
import random
import shutil
import numpy as np

def create_validation_dataset():

    index = 0
    for dir_name in os.listdir(Data.DATA_TRAINING):
        print('Index: ', index)
        index += 1
        #dir_train_path = Data.DATA_TRAINING2 + os.sep + dir_name
        #os.mkdir(dir_train_path)
        dir_validation_path = Data.DATA_VALIDATION + os.sep + dir_name
        os.mkdir(dir_validation_path)
        image_names = [image_name for image_name in os.listdir(Data.DATA_TRAINING + os.sep + dir_name)]
        random.shuffle(image_names)
        image_number = len(image_names)
        validation_number = image_number - 210
        #training_image_names = image_names[0:200]
        #for image_name in training_image_names:
        #    src = Data.DATA_TRAINING + os.sep + dir_name + os.sep + image_name
        #    dst = Data.DATA_TRAINING2 + os.sep + dir_name + os.sep + image_name
        #    shutil.copyfile(src, dst)
        validation_image_names = image_names[-validation_number:]
        for image_name in validation_image_names:
            src = Data.DATA_TRAINING + os.sep + dir_name + os.sep + image_name
            dst = Data.DATA_VALIDATION + os.sep + dir_name + os.sep + image_name
            shutil.move(src, dst)

def create_validation_dataset2():

    for dir_name in os.listdir(Data.DATA_TRAINING):
        dir_train_path = Data.DATA_TRAINING2 + os.sep + dir_name
        os.mkdir(dir_train_path)
        dir_validation_path = Data.DATA_VALIDATION2 + os.sep + dir_name
        os.mkdir(dir_validation_path)
        image_names = [image_name for image_name in os.listdir(Data.DATA_TRAINING + os.sep + dir_name)]
        random.shuffle(image_names)
        image_number = len(image_names)
        validation_number = image_number - 200
        training_image_names = image_names[0:200]
        for image_name in training_image_names:
            src = Data.DATA_TRAINING + os.sep + dir_name + os.sep + image_name
            dst = Data.DATA_TRAINING2 + os.sep + dir_name + os.sep + image_name
            shutil.copyfile(src, dst)
        validation_image_names = image_names[-validation_number:]
        for image_name in validation_image_names:
            src = Data.DATA_TRAINING + os.sep + dir_name + os.sep + image_name
            dst = Data.DATA_VALIDATION2 + os.sep + dir_name + os.sep + image_name
            shutil.copyfile(src, dst)


if __name__ == '__main__':
    create_validation_dataset()
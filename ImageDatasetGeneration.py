import os
import numpy as np
import cv2
import struct
import codecs
from datetime import datetime
import Utils
import time as t
import shutil

CHAR_LABEL_DICO_FILE_NAME = 'charLabelDicoFile.txt'
CASIA_DIR = 'E:\CHINESE_CHARACTER_RECOGNIZER\CASIA'

# this dir contains 240 .gnt files. Each .gnt file contains 3000 characters written by one writer.
GNT_TRAINING_PATH = CASIA_DIR + '\OFFLINE\HWDB1.1trn_gnt'

# this dir contains 60 .gnt files. Each .gnt file contains 3000 characters written by one writer.
GNT_TEST_PATH = CASIA_DIR + '\OFFLINE\HWDB1.1tst_gnt'

OUTPUT_DIR = CASIA_DIR + '\TEMP_GENERATED_DATASET'


# returns a map . key = chinese character, value = index
#  {'一' :0, '丁' : 1, '七' : 2, ...}
def build_char_index_dictionary():

    print("Building dictionary... ")
    start_time = t.time()
    dico_file = codecs.open(CHAR_LABEL_DICO_FILE_NAME, 'w', 'gb2312')
    character_set = set()
    for file_name in os.listdir(GNT_TRAINING_PATH):
        file_path = os.path.join(GNT_TRAINING_PATH, file_name)
        f = open(file_path, 'r')
        for _, tag_code in extract_image_and_tag_from_gnt_file(f):
            uni = struct.pack('>H', tag_code).decode('gb2312')
            character_set.add(uni)

    characters = list(character_set)
    character_index_dico = dict(zip(sorted(characters), range(len(characters))))
    for character, index in character_index_dico.items():
        dico_file.write(character + " " + str(index) + "\n")
    dico_file.close()
    print("Total %s characters. Execution time: %d s." % (str(len(character_index_dico)), t.time() - start_time))
    return character_index_dico

def create_validation_dataset():

    test_out_path = os.path.join(OUTPUT_DIR, "test")
    for dir_name in os.listdir(test_out_path):
        dir_validation_path = os.path.join(OUTPUT_DIR, "validation")
        if (not os.path.isdir (dir_validation_path) ):
            os.mkdir(dir_validation_path)
        image_names = [image_name for image_name in os.listdir(test_out_path + os.sep + dir_name)]
        # get files names starting by [241-270]
        filter_names = lambda file_name : file_name.startswith('24') or file_name.startswith('25') \
                                          or file_name.startswith('26') or file_name.startswith('270')

        validation_image_names = [file_name for file_name in image_names if filter_names(file_name)]
        for image_name in validation_image_names:
            src = test_out_path + os.sep + dir_name + os.sep + image_name
            dst_dir_path = dir_validation_path + os.sep + dir_name
            dst = dst_dir_path + os.sep + image_name
            if (not os.path.isdir(dst_dir_path)):
                os.mkdir(dst_dir_path)
            shutil.move(src, dst)

#  Extracts all the character images contained in one gnt file and put each extracted
# image into its corresponding directory.
def convert_gnt_to_png(gnt_dir, png_dir, char_label_dico, writer_index):

    start_time = datetime.now()
    gnt_number = len([name for name in os.listdir(gnt_dir) if os.path.isfile(name)])
    print("Number of .gnt file to process: %s ." % gnt_number)
    for gnt_file_name in os.listdir(gnt_dir):

        file_path = os.path.join(gnt_dir, gnt_file_name)
        gnt_file = open(file_path, "r")
        for image, tag_code in extract_image_and_tag_from_gnt_file(gnt_file):

            tag_code_uni = struct.pack('>H', tag_code).decode('gb2312') # chinese character
            character_dir = png_dir + "/" + '%0.5d' % char_label_dico[tag_code_uni]
            # character_dir examples : '00000', '00001', '00002'...
            # character_dir is a dir that contains all the 240 images of a given character
            os.makedirs(character_dir, exist_ok=True)
            image_name = '%0.3d' % writer_index + "_" + str(tag_code) + ".png"
            cv2.imwrite(character_dir + '/' + image_name, image)
        print("End processing file name: %s ." % gnt_file_name)
        gnt_file.close()
        writer_index += 1
    print("Execution time: %s ." % Utils.r(start_time))

def extract_image_and_tag_from_gnt_file(file):

    while True:
        header = np.fromfile(file, dtype="uint8", count=10)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        image = np.fromfile(file, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tag_code

def extract_images():

    training_out_path = os.path.join(OUTPUT_DIR, "training")
    test_out_path = os.path.join(OUTPUT_DIR, "test")
    char_label_dictionary = Utils.load_char_label_dico(CHAR_LABEL_DICO_FILE_NAME)

    print("Extracting training images.. ")
    writer_index = 1 # writer indices from training dataset belongs to [1,240]
    convert_gnt_to_png(GNT_TRAINING_PATH, training_out_path, char_label_dictionary, writer_index)

    print("Extracting test images.. ")
    writer_index = 241 # writer indices from test dataset belongs to [241,300]
    convert_gnt_to_png(GNT_TEST_PATH, test_out_path, char_label_dictionary, writer_index)

if __name__ == '__main__':
    extract_images()
    create_validation_dataset()

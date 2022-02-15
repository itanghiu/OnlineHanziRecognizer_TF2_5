import time
import os, glob
import logging
import tensorflow as tf
import numpy
import sys
import shutil
from datetime import datetime
import os.path
from os import path
import math
from PIL import Image
import numpy as np
import ImageDatasetGeneration
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks,optimizers, utils
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from TimeHistory import TimeHistory
from tensorflow.keras import Model
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from DisplayWeightCallback import DisplayWeightCallback
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import random
import matplotlib.pyplot as plt
import scipy
import time as t
import tensorflow.keras.models

#from BatchLogger import BatchLogger
import Utils
from Utils import get_label_char_dico
from Utils import get_class_size
import Csts
from Csts import VALIDATION_FREQ, DATA_TRAINING, DATA_VALIDATION, DATA_TEST, MODEL_DIR
from Utils import preprocess_image

logger = logging.getLogger('Cnn.py')
logger.setLevel(logging.INFO)

# builds the map whose keys are labels and values characters
label_char_dico = get_label_char_dico(ImageDatasetGeneration.CHAR_LABEL_DICO_FILE_NAME)
MODEL_NAME = 'hanzi_recog_model'
#STEPS_PER_EPOCH = 25
LEARNING_RATE = 5e-4  # 5e-3 #5e-2


class Cnn(Model):
    EVALUATION_STEP_FREQUENCY = 250  # Evaluates every 'evaluation_step_frequency' step
    SAVING_STEP_FREQUENCY = 180
    EPOCHS = 100  # Number of epoches
    RESTORE = True  # whether to restore from checkpoint
    RANDOM_FLIP_UP_DOWN = False  # Whether to random flip up down
    RANDOM_BRIGHTNESS = False  # whether to adjust brightness
    RANDOM_CONTRAST = True  # whether to random contrast
    GRAY = True  # whether to change the rbg to gray
    KEEP_NODES_PROBABILITY = 0.3
    WEIGHT_DECAY = 5E-4
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    mode = 'training' # Running mode: {"training", "test"}
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "{}/run-{}/".format(Csts.ROOT_LOG_DIR, now)

    def __init__(self, light_dataset = False):

        super(Cnn, self).__init__()
        self.training_path = DATA_TRAINING + '_LIGHT' if (light_dataset) else DATA_TRAINING
        self.validation_path = DATA_VALIDATION + '_LIGHT' if (light_dataset) else DATA_VALIDATION
        self.test_path = DATA_TEST + '_LIGHT' if (light_dataset) else DATA_TEST

        saved_model_path = os.path.join(MODEL_DIR, 'saved_model.pb')
        if os.path.exists(saved_model_path):
            print('LOADING EXISTING MODEL {} ..'.format(MODEL_DIR))
            self.model = tensorflow.keras.models.load_model(MODEL_DIR)
        else:
            training_class_size = get_class_size(self.training_path)
            print('Training class number: {}'.format(training_class_size))
            validation_class_size = get_class_size(self.validation_path)
            print('Validation class number: {}'.format(validation_class_size))

            self.model = self.build_graph(training_class_size)

    def get_run_logdir(self):

        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(Csts.ROOT_LOG_DIR, run_id)

    def training(self, light_dataset=False):

        start = t.time()
        print('Executing eagerly? ', tf.executing_eagerly())
        learning_rate = ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=2000, decay_rate=0.97, staircase=True)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        #optimizer = optimizers.SGD(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"], run_eagerly=True)
        #self.model.run_eagerly = True
        self.model.summary()

        #data_generator = ImageDataGenerator(rescale=1.0/255.0)
        #data_generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        #data_generator = ImageDataGenerator(rescale=2/ 255.0, featurewise_center=True, samplewise_center=True) # normalize to [-0.5,0.5]
        #data_generator = ImageDataGenerator(featurewise_center=True) # center around the mean computed on all the training set
        data_generator = ImageDataGenerator(preprocessing_function = preprocess_image)

        #images = self.get_images()
        #print('fitting images() ...')
        start = t.time()
        #data_generator.fit(images)
        #print('fitting images() took {:.2f} s'.format(t.time() - start))
        train_iterator = data_generator.flow_from_directory(self.training_path, color_mode='grayscale', target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), batch_size=Csts.BATCH_SIZE, class_mode='categorical')
        validation_iterator = data_generator.flow_from_directory(self.validation_path, color_mode='grayscale', target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), batch_size=Csts.BATCH_SIZE, class_mode='categorical')
        test_iterator = data_generator.flow_from_directory(self.test_path, color_mode='grayscale', target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE), batch_size=Csts.BATCH_SIZE, class_mode='categorical')

        # checkpoint_callback = ModelCheckpoint(Constants.MODEL + os.sep +MODEL_NAME+".h5", save_weights_only=False, save_freq='epoch')
        checkpoint_callback = ModelCheckpoint(MODEL_DIR, save_weights_only=False, save_freq=Cnn.SAVING_STEP_FREQUENCY)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        display_weight_callback = DisplayWeightCallback()
        #batch_logger = BatchLogger(validation_iterator)
        time_callback = TimeHistory()
        tensorboard_callback = TensorBoard(self.get_run_logdir(), update_freq='batch')
        #call_backs = [checkpoint_callback, tensorboard_callback, early_stopping_callback, display_weight_callback]
        call_backs = [checkpoint_callback, tensorboard_callback, early_stopping_callback]

        batchX, batchy = train_iterator.next()
        print('Train Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        #print('batchy shape={}'.format(batchy.shape))
        batchX, batchy = validation_iterator.next()
        print('Val Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

        training_size = Utils.get_size(self.training_path)
        validation_size = Utils.get_size(self.validation_path)
        print('Training size: ', training_size)
        print('validation size: ', validation_size)
        print('len(train_iterator): ', len(train_iterator))

        BATCH_PER_EPOCH = math.ceil(training_size / Csts.BATCH_SIZE)
        #BATCH_PER_EPOCH =12
        VALIDATION_BATCH = math.ceil(validation_size / Csts.BATCH_SIZE)
        print(' BATCH_PER_EPOCH:', BATCH_PER_EPOCH)
        print(' VALIDATION_BATCH:', VALIDATION_BATCH)
        self.history = self.model.fit_generator(train_iterator, validation_data=validation_iterator,
                                 #steps_per_epoch=BATCH_PER_EPOCH,
                                 steps_per_epoch=len(train_iterator),
                                 validation_steps=VALIDATION_FREQ,
                                 epochs=Cnn.EPOCHS,
                                 callbacks=call_backs,
                                 verbose = 1)
        self.model.save(Csts.MODEL_DIR)
        test_size = Utils.get_size(self.test_path)
        STEPS_TEST = math.ceil(test_size / Csts.BATCH_SIZE)
        score = self.model.evaluate(test_iterator, steps=STEPS_TEST)
        print('Metrics names:', self.model.metrics_names)
        print('Score:', score)
        print('Training took {:.2f} s'.format(t.time()-start))
        self.draw_history()

    def draw_history(self):

        print(self.history.history.keys())
        #  "Accuracy"
        plt.plot(self.history.history['accuracy'])
        val_accuracy = self.history.history['val_accuracy']
        #plt.plot(val_accuracy)
        n = VALIDATION_FREQ
        #plt.plot([x + n - 1 for x in range(0, len(val_accuracy), n)], nth_value_plot, "o")
        epochs = len(val_accuracy)
        plt.plot([x for x in range(0, epochs, n)], val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('Batch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # "Loss"
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def draw_history2(self):

        plt.style.use('seaborn-whitegrid')
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def get_validation_image_sample(self):

        directories = os.listdir(self.validation_path)
        images = []
        for dir in directories:
            files = os.listdir(self.validation_path + os.sep + dir)
            nbr_files = len([name for name in files]) - 1
            random_index = random.randint(0, nbr_files)
            random_image = files[random_index]
            image = Utils.resize_image(self.validation_path + os.sep + dir + os.sep + random_image)
            images.append(image)
        return images

    def get_test_image_sample(self):

        directories = os.listdir(self.test_path)
        images = []
        for dir in directories:
            files = os.listdir(self.test_path + os.sep + dir)
            nbr_files = len([name for name in files])-1
            random_index = random.randint(0, nbr_files)
            random_image = files[random_index]
            image = Utils.resize_image(self.test_path + os.sep + dir + os.sep + random_image)
            images.append(image)
        return images

    def get_images(self):

        print('get_images() ...')
        start = t.time()
        directories = os.listdir(self.training_path)
        images = []
        for dir in directories:
            files = os.listdir(self.training_path + os.sep + dir)
            nbr_files = len([name for name in files])-1
            random_index = random.randint(0, nbr_files)
            random_image = files[random_index]
            image = Utils.resize_image(self.training_path + os.sep + dir + os.sep + random_image)
            images.append(image)
        print('get_images() took {:.2f} s'.format(t.time() - start))
        return images

    def evaluate_against_training(self):

        #data_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        data_generator = ImageDataGenerator(preprocessing_function = preprocess_image)
        validation_iterator = data_generator.flow_from_directory(self.validation_path, color_mode='grayscale',
                                                                 target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE),
                                                                 batch_size=Csts.BATCH_SIZE, class_mode='categorical')
        reconstructed_model = keras.models.load_model("hanzi_recog_model")
        validation_size = Utils.get_size(self.validation_path)
        STEPS_VALIDATION = math.ceil(validation_size / Csts.BATCH_SIZE)
        score = reconstructed_model.evaluate(validation_iterator, steps=STEPS_VALIDATION)
        print('Metrics names:', self.model.metrics_names)
        print('Score for validation dataset:', score)

    def evaluate_against_validation(self):

        #data_generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        data_generator = ImageDataGenerator(preprocessing_function = preprocess_image)
        images = self.get_validation_image_sample()
        data_generator.fit(images)
        validation_iterator = data_generator.flow_from_directory(self.validation_path, color_mode='grayscale',
                                                                 target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE),
                                                                 batch_size=Csts.BATCH_SIZE, class_mode='categorical')
        reconstructed_model = keras.models.load_model("hanzi_recog_model")
        validation_size = Utils.get_size(self.validation_path)
        STEPS_VALIDATION = math.ceil(validation_size / Csts.BATCH_SIZE)
        score = reconstructed_model.evaluate(validation_iterator, steps=STEPS_VALIDATION)
        print('Metrics names:', self.model.metrics_names)
        print('Score for validation dataset:', score)

    def evaluate_against_test(self):

        #data_generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        data_generator = ImageDataGenerator(preprocessing_function = preprocess_image)
        images = self.get_test_image_sample()
        data_generator.fit(images)
        test_iterator = data_generator.flow_from_directory(self.test_path, color_mode='grayscale',
                                                           target_size=(Csts.IMAGE_SIZE, Csts.IMAGE_SIZE),
                                                           batch_size=Csts.BATCH_SIZE, class_mode='categorical')
        reconstructed_model = keras.models.load_model("hanzi_recog_model")
        test_size = Utils.get_size(self.test_path)
        STEPS_TEST = math.ceil(test_size / Csts.BATCH_SIZE)
        score = reconstructed_model.evaluate(test_iterator, steps=STEPS_TEST)
        print('Metrics names:', self.model.metrics_names)
        print('Score:', score)


    def build_graph(self, class_size):

        self.model = models.Sequential()
        input_shape = (Csts.IMAGE_SIZE, Csts.IMAGE_SIZE, 1)
        self.conv3_1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='he_uniform')
        self.max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3_2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform')
        self.max_pool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3_3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform')
        self.max_pool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3_4 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform')
        self.max_pool_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.flatten = Flatten()
        #self.fc1 = Dense(1024, activation='relu', kernel_regularizer=l2(Cnn.WEIGHT_DECAY))
        self.fc1 = Dense(1024, activation='relu')
        self.dropout1 = Dropout(Cnn.KEEP_NODES_PROBABILITY)
        #self.fc2 = Dense(class_size, activation='relu', kernel_regularizer=l2(Cnn.WEIGHT_DECAY))
        self.fc2 = Dense(class_size, activation='relu')
        self.dropout2 = Dropout(Cnn.KEEP_NODES_PROBABILITY)
        self.logits = Dense(class_size, activation='softmax')

        self.model.add(self.conv3_1)
        self.model.add(BatchNormalization())
        self.model.add(self.max_pool_1)

        self.model.add(self.conv3_2)
        self.model.add(BatchNormalization())
        self.model.add(self.max_pool_2)

        self.model.add(self.conv3_3)
        self.model.add(BatchNormalization())
        self.model.add(self.max_pool_3)

        self.model.add(self.conv3_4)
        self.model.add(BatchNormalization())
        self.model.add(self.max_pool_4)

        self.model.add(self.flatten)
        self.model.add(self.fc1)
        self.model.add(BatchNormalization())
        self.model.add(self.fc2)
        self.model.add(BatchNormalization())
        self.model.add(self.logits)
        return self.model


    def load_model(self):
        print('Loading model...')
        savedmodel_dir = Csts.MODEL + os.sep + MODEL_NAME + '.h5'
        self.model.load_weights(savedmodel_dir) # Restore the model's state
        print('End Loading model.')

    def save_model(self):

        # --- Saving model
        logger.info('Saving model %s...'%(MODEL_NAME))
        savedmodel_dir = Csts.MODEL + os.sep + MODEL_NAME + '.h5'
        # delete the model directory
        if (path.exists(savedmodel_dir)):
            try:
                shutil.rmtree(savedmodel_dir)
            except OSError as e:
                print("Error: %s : %s" % (savedmodel_dir, e.strerror))

        # Save weights to a HDF5 file
        #self.model.save_weights(savedmodel_dir + '.h5', save_format='h5')
        self.model.save_weights(savedmodel_dir, save_format='h5')
        logger.info('End saving model.')

    def recognize_image(self, image_file_name):

        reconstructed_model = keras.models.load_model(Csts.MODEL_DIR)
        image = Utils.prepare_image2(image_file_name)
        predicted_probabilities = reconstructed_model.predict(image)
        predicted_chars, probabilities, indices = Cnn.get_characters_from_proba(predicted_probabilities)
        return predicted_chars, probabilities, indices


    def recognize_image_with_tf_lite(image_file_name):

        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=Csts.TF_LITE_MODEL)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        #image = Utils.prepare_image(image_file_name)
        image = Utils.prepare_image2(image_file_name)
        #image = np.expand_dims(image, axis=0)
        image = np.float32(image)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        predicted_probabilities = interpreter.get_tensor(output_details[0]['index'])
        predicted_chars, probabilities, predicted_indices = Cnn.get_characters_from_proba(predicted_probabilities)
        return predicted_chars, probabilities, predicted_indices

    def get_characters_from_proba(predicted_probabilities):

        predicted_indexes = np.argsort(predicted_probabilities)
        predicted_indexes = np.squeeze(predicted_indexes) # removes first dimension
        predicted_indexes = predicted_indexes[::-1] # reverses the array to have the higest proba first
        predicted_indexes = predicted_indexes[0:Csts.FIRST_TOP_CHARACTERS]
        #real_label_dict = Utils.get_real_labels()
        #real_indexes = [real_label_dict.get(k) for k in predicted_indexes]
        predicted_chars = [label_char_dico.get(index) for index in predicted_indexes]
        probabilities = np.sort(predicted_probabilities)
        probabilities = np.squeeze(probabilities) # removes first dimension
        probabilities = probabilities[::-1]  # reverses the array to have the higest proba first
        probabilities = probabilities[0:Csts.FIRST_TOP_CHARACTERS]
        return predicted_chars, probabilities, predicted_indexes


    @staticmethod
    def convert_to_tensor_lite():
        print('Converting model located at ' + Csts.MODEL_DIR)
        converter = tf.lite.TFLiteConverter.from_saved_model(Csts.MODEL_DIR)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)

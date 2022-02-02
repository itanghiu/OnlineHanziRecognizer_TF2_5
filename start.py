import Cnn
import sys
import tensorflow as tf
from Cnn import Cnn
import Csts
import time as t

if __name__ == "__main__":  # Learning mode

    arg = sys.argv[1:]
    mode = arg[0].split('=')[1]
    print("Mode: %s " % mode)
    print('Python version:', sys.version)
    print('Tensorflow version:', tf.version.VERSION)
    print(tf.executing_eagerly())
    var = tf.Variable([3, 3])
    #if tf.test.is_gpu_available():
    if tf.config.list_physical_devices('GPU'):
        print('Running on GPU')
        #print('GPU #0?')
        #print(var.device.endswith('GPU:0'))
    else:
        print('Running on CPU')

    if mode == 'evaluate_against_training':
        cnn = Cnn()
        cnn.evaluate_against_training()

    if mode == 'evaluate_against_validation':
        cnn = Cnn()
        cnn.evaluate_against_validation()

    if mode == 'evaluate_against_test':
        cnn = Cnn()
        cnn.evaluate_against_test()

    if mode == 'recognize_image':
        cnn = Cnn()
        start = t.time()
        image_file_names = ['images/bu4.png', 'images/er2.png', 'images/qi1.png', 'images/shi3.png', 'images/xia.png', 'images/yi1.png']
        for image_file_name in image_file_names:
            characters, probabilities, indices = cnn.recognize_image(image_file_name)
            print('Predicting character {}'.format(image_file_name.split('/')[1]))
            for i in range(Csts.FIRST_TOP_CHARACTERS):
                print('Probability: {:.2f}  Character: {}'.format(float(probabilities[i]), characters[i]))
        print('Inference done in {:.2f} s.'.format(t.time() - start))

    if mode == 'recognize_image_with_tf_lite':
        start = t.time()
        image_file_names = ['images/bu4.png', 'images/er2.png', 'images/qi1.png', 'images/shi3.png', 'images/xia.png',
                            'images/yi1.png']
        for image_file_name in image_file_names:
            print('Predicting character {}'.format(image_file_name.split('/')[1]))
            characters, probabilities, indices = Cnn.recognize_image_with_tf_lite(image_file_name)
            for i in range(Csts.FIRST_TOP_CHARACTERS):
                print('Probability: {:.2f}  Index: {} Character: {}'.format(float(probabilities[i]), indices[i], characters[i]))
        print('Inference done in {:.2f} s.'.format(t.time()-start))

    elif mode == "training":
        light_dataset = False
        cnn = Cnn(light_dataset)
        cnn.training(light_dataset)

    elif mode == "convert_to_tensor_lite":
        cnn = Cnn(False)
        cnn.convert_to_tensor_lite()

else:  # webServer mode
    print('Web server mode')

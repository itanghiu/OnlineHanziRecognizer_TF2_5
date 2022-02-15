from tensorflow.keras.callbacks import Callback
import Csts


class DisplayWeightCallback(Callback):

    def on_batch_end(self, batch, logs={}):

        #for layer_i in range(len(self.model.layers)):
        w1 = self.model.get_layer('conv2d').get_weights()[0]
        w7 = self.model.get_layer('conv2d_3').get_weights()[0]
        w9 = self.model.get_layer('dense').get_weights()[0]
        #print('\nw1 {:.2f} {:.2f} {:.2f}'.format(w1[0][0][0][0], w1[0][0][0][1], w1[0][0][0][2]))
        print('\nw7 {:.2f} {:.2f} {:.2f}'.format(w7[0][0][0][0], w7[0][0][0][1], w7[0][0][0][2]))
        print('w9 {:.5f} {:.5f} {:.5f}'.format(w9[0][0], w9[0][1], w9[0][2]))

# OnlineHanziRecognizer

This project is an implementation of the paper "Deep Convolutional Network for Handwritten Chinese Character Recognition" by Yuhao ZHUANG.
It is based on a convolutional neural network (CNN)

### Requirements :
Python version: 3.6.3

Tensorflow version: 2.5.0

### Installing the dependencies :

```
pip install -r requirements.txt
```

## Project options

### Generate the training and test dataset:

The dataset used is the HWDB1.1 dataset. 
This dataset contains 3,755 Chinese characters and 171 alphanumeric
and symbols. Each class of character is represented by 300 handwritten images drawn by 300 writers. 
Each writer has written the 3 755 Chinese characters. 

To download it :
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

In ImageDatasetGeneration.py, set the CASIA_DIR variable to the path where you installed the dataset.

``` python ImageDatasetGeneration.py ```

### Start training:

``` python start.py --mode=training ```

To visualize metrics such as loss and accuracy using Tensorboard :

``` tensorboard --logdir=./logs/ --port=8090 --host=127.0.0.1 ```
 
In a browser , go to : http://localhost:8090/

### Launching the Web server :

The project comes with a simple web user interface for drawing characters with the mouse.
The interface displays the recognized characters, as the character is drawn.

``` python app.py ``` 

In a browser , go to : http://localhost:5000/

### Recognizing images 

``` python  start.py --mode=recognize_image ```

### Convert the model to TensorLite

``` python  start.py --mode=convert_to_tensor_lite ```

## The model training

The implemented model corresponds to the M6 model as defined in the paper.
Training took 2 days on a Dell XPS (16 Go, Nvidia GeForce GTX 1050 with 4 Gb) and accuracy on validation set is 93.2 %.








 



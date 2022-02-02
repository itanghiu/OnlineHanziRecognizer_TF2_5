## OnlineHanziRecognizer

### Requirements :
Python version: 3.6.3
Tensorflow version: 1.14.0

> Install Python 3.6.3
> pip3 install --upgrade tensorflow==1.14.0


#### Generate the training and test dataset:

The dataset used is the HWDB1.1 dataset. To download it :
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

In ImageDatasetGeneration.py, set the CASIA_DIR variable to the path where you installed the dataset
> python ImageDatasetGeneration.py

#### Start training:

 Script path: PATH_TO_PROJECT\OnlineHanziRecognizer_tf1_14_0\start.py
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer_tf1_14_0
> python start.py --mode=training

#### Launching the Web server :

- > pip install tensorflow==1.13.2
- > pip install Flask
- > pip install Image
- > pip install opencv-python
- > 

 Script path : PATH_TO_PROJECT\OnlineHanziRecognizer_tf1_14_0\app.py
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer_tf1_14_0
> python app.py
In a browser , go to : http://localhost:5000/

#### Start tensorBoard:

tensorboard --logdir=./logs/ --port=8090 --host=127.0.0.1
In a browser , go to : http://localhost:8090/

 



# cifar100 classifier using Tensorflow
### introduction:
* cifar 100 is a dataset of  of 100 different classes, it's divided into 50000 training data and 10000 test data.
* the goal is to develop a classifier using Convolutional nural network which is able to achive high accuracy on test data.
### about:
* the project is divided into 2 parts :
  * bulinding the model and the training operation  handeled by **classifier.py**
  * running the code and getting the results handeled by **pipline.py**
* to change the parameters of the training like batch size or learninig rate you edit that in **config.py** file
### installation:
first clone the repo and got to the directory
```bash
git clone https://github.com/AhmedGhazale/cifar100-classifier.git
cd cifar100-classifier
```
#### Dependencies

to be able to run the code you need to install these libraries:

* lxml==4.2.5
* matplotlib==2.2.2
* numpy==1.15.4
* opencv-python==3.4.0.12
* pandas==0.23.4
* Pillow==5.1.0
* protobuf==3.6.1
* scikit-image==0.14.2
* scikit-learn==0.19.2
* scipy==1.0.1
* six==1.12.0
* sklearn==0.0
* tensorboard==1.11.0
* tensorflow-gpu==1.11.0
* tensorflow-tensorboard==0.4.0

To install required dependencies run:
```bash
pip install -r requirements.txt
```
to start the training simply run
```bash
python3 piplin.py
```
if you want to modify the learning rate or to load a pretrained model edit that in **config.py**


### results:
* final model achived accuracy of **57%** on the test set
* you can download the pretrained model from https://drive.google.com/open?id=19DCFqJBudXUFh8BTUYCu0SYYwtJKdFMP





### Medical Image Captioning



This is the code for the submission for the Medical [Image Captioning Challenge](https://www.imageclef.org/2022/medical/caption). 

This project was done by a team of Big Data and Text Analytics Msc Student at the university of Essex as part of the group project course.


### Our Solution


_Nothing is said that has not been said before._ Terence (ca 195 - 159 BC ) The Eunuch Prol.

Solve this challenge we used and Encoder Decoder Architecture.

#### Encoder 

The encoder consist of a pretrained reset50 model to which we removed the final layers and add a batch normalization to return a feature vector of size 512 for each image.

#### Decoder

The decoder is a 8 layers LSTM that takes the images representation and the and predict the corresponding captions.


### Training 

We trained the model using 30 epoch, and using the cross entropy loss .

The result yield a bleu score of 0.2.  on our validation set.

### Instructions to reproduce the results


#### Installing packages: 


We have used Python3.8.5 and [poetry](https://python-poetry.org/docs/) to install the packages and the dependencies.

If you have poetry installed you can simply run the following command:

- `poetry shell ` 
To create the project virtual environment.

Run : 

- `poetry install`



#### Download the dataset.

The dataset is large and we used [dagshub](https://dagshub.com/) and [dvc](https://dvc.org/) to version control it.

- Add the remote with the following command:

`dvc remote add origin https://dagshub.com/espoirMur/image-clef-2022-essex-submission.dvc`


- Pull the dataset from the remote repository using the following command : 

`dvc pull -r origin`


Go grab a coffee and wait for the dataset to be downloaded.

Check the folder `data/raw/training-images` for the training images and `data/raw/validation-images` for the validation images.

The corresponding captions are in the `data/raw/caption-prediction` folders.

### Setup MLFLOW to reproduce the experiments:


We used the [MLflow](https://www.mlflow.org/) to reproduce the experiments.

You need to have a daghubs account or mlflow account to run the experiments.
Once you have let say a dagshub account you can run the following command to log your experiments:


```
MLFLOW_TRACKING_URI=your-daghubs-url\
MLFLOW_TRACKING_USERNAME=username-from-dagshub \
MLFLOW_TRACKING_PASSWORD=your_token \
```
#### Training the model


The code for training the model is saved under `src/models/train_model.py` check it out and edit the hyperparameters to your liking.


If everything is okay for you then run the following command:

`python src/models/train_model.py` and wait for the model to train.



### Did I miss something?: 

Please let us know , if you have any other questions or suggestions.


### References: 

90 % of the code was copied from the internet. Here are the majors references : 

- [Image Pytorch ](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [ Image captioning Udacity Challenge](https://github.com/rammyram/image_captioning)
- [Medical Image Captioning](https://towardsdatascience.com/medical-image-captioning-on-chest-x-rays-a43561a6871d)
- [Captioning Blog](http://shikib.com/captioning.html)

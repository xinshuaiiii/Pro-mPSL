# Pro-mPSL/Datastes
The Datastes you can find in Fesc-PSL/Datasets.

# ProtT5
You can download the ProtT5-XL-UniRef50 model from this [[website]](https://github.com/agemagician/ProtTrans),there is a detailed tutorial on how to use the model.If you download the model locally,ProtT5.py can help you convert sequences to embeddings,you just need to modify the model path in the file.
python generate_embeddings.py


# train
requirements.txt is the environment needed for running.


Model of Gram-positive bacteria is saved in Pos/Gram_train_model.py , The hyperparameters have been set according to the parameters in the paper. 

Training file is saved in Pos/Gram+_train.py , you need to convert the training set and validation set into embeddings through the ProtT5 model and set them on train_data and val_data. Model of Gram-positive bacteria is same.

There is a demo for test.  demo/test-152-single-del-label and demo/test-152-single-del-seq is a simple dataset. Run ProtT5.py to convert test-152-single-del-seq to embeddings,and transform it into training set and test set.Place the training data and val data in the path of Pos/Gram+_train.py , you can start to train and get the training model.

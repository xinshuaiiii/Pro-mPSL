# Pro-mPSL/Datastes
The Datastes you can find in 'Fesc-PSL/Datasets'.

# ProtT5
You can download the ProtT5-XL-UniRef50 model from this [[website]](https://github.com/agemagician/ProtTrans),there is a detailed tutorial on how to use the model.If you download the model locally,'ProtT5.py' can help you convert sequences to embeddings,you just need to modify the model path in the file.

Configure the model and sequence files, that is, put the pre-trained T5 model in the '../prot5' directory.Put your protein sequence in the 'sequence.txt file'.Run the following command.
```
pip install -r requirements.txt
python Prott5.py --model_path ../prot5 --filename sequence.txt --output_path embeddings.npy
```


# train
'requirements.txt' is the environment needed for running.

Model of Gram-positive bacteria is saved in 'Pos/Gram_train_model.py' , The hyperparameters have been set according to the parameters in the paper. 

Training file is saved in 'Pos/Gram+_train.py' , you need to convert the training set and validation set into embeddings through the ProtT5 model and set them on 'train_data' and 'val_data'. Model of Gram-positive bacteria is same.

There is a demo for test.  'demo/test-152-single-del-label' and 'demo/test-152-single-del-seq 'is a simple dataset. 
Run ProtT5.py to convert 'test-152-single-del-seq' to embeddings.
```
python Prott5.py --model_path ../prot5 --filename test-152-single-del-seq --output_path embeddings.npy
```
Then transform it into training set and test set.Place the training data and val data in the path of 'Pos/Gram+_train.py' , you can start to train and get the training model.
```
python Gram+_train.py --train_data_path train.pkl --val_data_path val.pkl --source_dir your_dir --log_dir /path/to/your/logs --prefix train_new
```
Neg is the same.

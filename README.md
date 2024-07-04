# Pro-mPSL/Datastes
The Datastes you can find in https://github.com/xinshuaiiii/FESC-PSL/tree/main/dataset.

# Creating a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies.The command is as follows：
```
conda create -n predict pyhton=3.7.13
conda activate predict
git clone https://github.com/xinshuaiiii/Pro-mPSL.git
cd Pro-mPSL
pip install -r requirements.txt
```

# ProtT5
You can download the ProtT5-XL-UniRef50 model from this [[website]](https://github.com/agemagician/ProtTrans),there is a detailed tutorial on how to use the model.
If you download the model locally,'ProtT5.py' can help you convert sequences to embeddings,you just need to modify the model path in the file.
Configure the model and sequence files, that is, put the pre-trained T5 model in the '../prot5' directory.Put your protein sequence in the 'sequence.txt file'.Run the following command.
```
python Prott5.py --model_path ../prot5 --filename sequence.txt --output_path embeddings.npy
```

# Predict

Model of Gram-positive bacteria is saved in 'Pos/Gram_train_model.py' , The hyperparameters have been set according to the parameters in the paper. 
Training file is saved in 'Pos/Gram+_train.py' , you need to convert the training set and validation set into embeddings through the ProtT5 model and set them on 'train_data' and 'val_data'. Model of Gram-positive bacteria is same.


There is a demo for test.  'demo/test-152-single-del-label' and 'demo/test-152-single-del-seq 'is a simple dataset. 
Run ProtT5.py to convert 'test-152-single-del-seq' to embeddings.
```
cd /your/path
python Prott5.py --model_path ../prot5 --filename test-152-single-del-seq --output_path embeddings.npy
```
Then transform it into training set and test set.Place the training data and val data in the path of 'Pos/Gram+_train.py' , you can start to train and get the training model.If your training and validation data files are located at data/train_data.pkl and data/val_data.pkl respectively, and you want to save the model and logs to the results/ directory, using the prefix experiment1, you can run:
```
cd /your/path/
python Gram+_train.py --train_data_path data/train_data.pkl --val_data_path data/val_data.pkl --source_dir results --prefix experiment1 --total_epoch 50 --batch_size 64 --learning_rate 0.000766
```
Neg is the same.
```
cd /your/path/
python Gram-_train.py --train_data_path data/train_data.pkl --val_data_path data/val_data.pkl --source_dir results --prefix experiment1 --total_epoch 50 --batch_size 64 --learning_rate 0.000766
```

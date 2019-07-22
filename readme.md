# Using Bert for classification problem

Note: I am using Ubuntu 16.x+ for this problem. Bert runs on Tensorflow and at the time of this project, Tensorflow only worked up to Python 3.6 

## Set Up steps
Steps to get the set up. The input data is available in `input_data` directory

With python 3 and pip installed:

1. Make a directory for the project `mkdir bert_classify && cd bert_classify`
2. Clone the bert directory from github `git clone git@github.com:google-research/bert.git`. 
3. Create a new virtual environment and enter it `python3 -m venv bert_linux`
4. Make output repo to add files after running model `mkdir output_data`
5. Install modules requred on top of tensor flow for this project `pip install -r requirements.txt`. It includes tensorflow w/ dependencies, pandas w/dependencies, scikit-learn w/dependencies, and jupyter labs

## Operational Steps

All the steps are available in the [`prep_for_bert.ipynb`](prep_for_bert.ipynb) notebook. The idea first to get the data in shape for bert to train on, then use the command line to run the training and prediction, and use the results to get the predictions in the format we want to present in. 


1. Create the dataset required for bert to create the input data required to run the model.
2. Run the BERT provided classification script and provide the flags as required:
    ```
    python3 bert\run_classifier.py \
    --task_name= cola \
    --do_train=true \
    --do_eval=true \
    --data_dir=input_data \ 
    --vocab_file=bert/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=bert/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=50 \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=output_data \ 
    ```
3. Run bert predictor with the flag `--do_predict=true` and remove any training flags
    ```
    python bert/run_classifier.py \
    --task_name=cola \
    --do_predict=true \
    --data_dir=input_data \
    --vocab_file=bert/uncased_L-12_H-768_A-12/vocab.txt \ --bert_config_file=bert/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=bert/uncased_L-12_H-768_A-12/bert_model.ckpt \ --max_seq_length=128 \
    --output_dir=output_data
    ```
4. Use the predict results, the probabilites of each labels for each observation, and align them with the test data. 

Data probably gathered from sf city council meeting minutes [example](https://sfbos.org/sites/default/files/bag062519_minutes.pdf)
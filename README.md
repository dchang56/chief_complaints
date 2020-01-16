# Classifying free-text chief complaints in the Emergency Department using BERT

## what are chief complaints and why are they important

## what did we do

We trained a clinical BERT model 1.8 million visits from 2013-2018 in the Yale hospital system to map free-text chief complaints to structured categories of presenting problems.

## How? brief summary of training

train.py in training folder. If you have any questions, email me

## Pretrained models

Download pretrained models and accompanying files from: https://drive.google.com/drive/folders/13GpWRlJCt8Sv_8fpR1nLkIGiIn4aFuAO?usp=sharing

There are 4 options for pretrained models based on the different versions of training data (subsets with 29, 117, 260, and 434 possible labels). Each folder contains 4 files:

`label_map.json`: a dictionary mapping labels to integers.

`pytorch_model.bin`: pretrained pytorch model.

`training_args.bin`: a dictionary of training args used for the model.

`vocab.txt`: vocab used by tokenizer.

`config.json`: configuration file for the model.

## Generate predictions for your own chief complaint data

clone repo
pip install -r requirements.txt
download pretrained model

To run the model on your own set of chief complaints, it should just be a simple text file with each row being a chief complaint. A sample input file `test_input.txt` is provided.

To run predict.py, use the following template:

```bash
export INPUT_FILE=/path/to/input_file
export MODEL_PATH=/path/to/downloaded/folder
export OUTPUT_DIR=/path/to/output_dir

python predict.py \
    --input_file=$INPUT_FILE \
    --model_path=$MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --label_map=$MODEL_PATH/label_map.json \
    --config_name=$MODEL_PATH/config.json \
    --tokenizer_name=$MODEL_PATH/vocab.txt \
    --k=5 \
    --do_lower_case
```

The following files will be saved to the output directory:

`cached_data`: a cache of processed input data for convenience.

`prediction_labels.csv`: a csv file of top k predicted labels for each chief complaint (same order as input file).

`prediction_probs.csv`: a csv file of corresponding probability scores for the labels.

`output.txt`: a text file with user friendly printouts of the results.



a text file containing printouts of the chief complaints and their top predictions with their scores, two csv files containing the predicted labels and their probabilities, and a text file containing user-friendly printouts of the chief complaints and their top predictions along their scores,

# Classifying free-text chief complaints in the Emergency Department using BERT

Repository for the paper Generating Contextual Embeddings for Emergency Department Chief Complaints using BERT (submitted to JAMIA)

A clinical BERT model was trained on 1.8 million emergency department chief complaints to classify free-text chief complaints to provider-assigned labels (also known as presenting problems). 

This repository contains the code used for the project. Most notably, we provide a script `predict.py` and instructions for generating predictions for custom chief complaint datasets using our pretrained models.

## Pretrained models

Download the pretrained models and accompanying files from: https://drive.google.com/drive/folders/13GpWRlJCt8Sv_8fpR1nLkIGiIn4aFuAO?usp=sharing

Use the following command to extract the archive:

`tar -xzvf [archivename]`

There are 4 versions for pretrained models based on the different subsets of training data used (subsets with 29, 117, 260, and 434 most frequently occuring labels). Each folder contains 5 files:

`label_map.json`: a dictionary mapping labels to integers.

`pytorch_model.bin`: pretrained pytorch model.

`training_args.bin`: a dictionary of training args used for the model.

`vocab.txt`: vocab used by tokenizer.

`config.json`: configuration file for the model.

Which version to use depends on the user's preference for specificity of the labels. Check label_map.json to see if the label space is sufficiently comprehensive for your purpose. 

## Generate predictions for your own chief complaint data

Follow these steps:

1. `git clone https://github.com/dchang56/chief_complaints && cd chief_complaints`

2. `pip install -r requirements.txt`

3. Download and extract pretrained model archive from the link above

4. Prepare your input file as a simple text file with one chief complaint per line (`test_input.txt` is provided as an example)

5. run `predict.py` using the following template with appropriate paths. The argument `k` allows you to get the top k predictions of the model for each chief complaint.

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


## Contact

Please post an issue or email david.chang@yale.edu if you have any questions.

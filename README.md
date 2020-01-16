# Classifying free-text chief complaints in the Emergency Department using BERT

## description of project
1.8 million visits in yale hospital system


Download pretrained models and accompanying files from: https://drive.google.com/drive/folders/13GpWRlJCt8Sv_8fpR1nLkIGiIn4aFuAO?usp=sharing

There are 4 options for pretrained models based on the different versions of training data (subsets with 29, 117, 260, and 434 possible labels). Each folder contains 4 files:

`label_map.json`: a dictionary mapping labels to integers.

`pytorch_model.bin`: pretrained pytorch model.

`training_args.bin`: a dictionary of training args used for the model.

`vocab.txt`: vocab used by tokenizer.

## Usage

If you want to run the model on your own set of chief complaint texts, it should just be a simple text file with each row being a chief complaint text. A sample input file is provided `test_input.txt`

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
    --do_lower_case
```

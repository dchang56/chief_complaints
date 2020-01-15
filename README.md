# Classifying free-text chief complaints in the Emergency Department using BERT


Download pretrained models and accompanying files from: https://drive.google.com/drive/folders/13GpWRlJCt8Sv_8fpR1nLkIGiIn4aFuAO?usp=sharing

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

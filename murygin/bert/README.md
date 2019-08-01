### Running Actions

All actions are done in current directory.

1. Download uncased BERT-Base:
   ```
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   unzip uncased_L-12_H-768_A-12.zip
   ```
2. Download github repository with SNIPS dataset: `git clone https://github.com/snipsco/nlu-benchmark.git`.
3. Change serving path in function `export_saved_model` on line 896 in file `run_bert_nlu.py`.
   This path will store model in savedmodel format for later conversion to tflite format.
4. Comment or not functions `create_training_graph` and `create_eval_graph` on lines 534 and 546 in file `run_bert_nlu.py`.
5. If function `create_training_graph` isn't commented than you can change or exclude second argument `quant_delay`.
   This argument sets the number of steps after which quantization simulating will begin.
6. Run `run_bert_nlu.py` as it described in comments in the beginning of file. For example:
   ```
   python3 run_bert_nlu.py \
        --do_train \
        --data_dir=./nlu-benchmark \
        --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
        --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json  \
        --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
        --max_seq_length=48 \
        --train_batch_size=16 \ 
        --learning_rate=2e-5 \ 
        --num_train_epochs=10 \ 
        --output_dir=./output \
        --save_checkpoints_steps=200
    ```
7. Run `tflite_convert --output_file=./bert.tflite --saved_model_dir=PATH_TO_SAVEDMODEL` to convert TF model to TFLite format.

### Notes

* TensorFlow BERT isn't converted in TFLite model because of error related with permutation: <https://github.com/tensorflow/tensorflow/issues/22109>.
* Base BERT isn't fine-tuned after inserting fake quantization at the first step. It is necessary to use "quant\_delay" parameter to fine-tune BERT.
* Used a lot of GPU memory when setting parameter "quant\_delay". If GPU memory isn't enough then you can reduce number of transformer layers in the configuration file.

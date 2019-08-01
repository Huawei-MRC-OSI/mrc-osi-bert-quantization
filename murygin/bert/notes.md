* TensorFlow BERT isn't converted in TFLite model because of error related with permutation: <https://github.com/tensorflow/tensorflow/issues/22109>
* Base BERT isn't learned after inserting fake quantization. When using quant delay there aren't enough GPU memory.
* BERT with 5 layers is learned after inserting fake quantization and after using quant delay.
* Main question now: will base BERT be learned after using quant delay?
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  inp0 = tf.placeholder('int32', [10, 10])
  res = tf.reshape(inp0, [100])
  res = tf.one_hot(res, depth=10)

  print(res.shape)

  #print(tf.get_default_graph().as_graph_def())

  def representative_dataset_gen():
    for i in range(10):
      yield [np.random.random_integers(0, 9, size=inp0.get_shape()).astype('int32')]

  converter = tf.lite.TFLiteConverter.from_session(sess, [inp0], [res])
  converter.representative_dataset = representative_dataset_gen
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  tflite_model = converter.convert()
  model_file = "/tmp/tflite_one_hot_bug.tflite"
  open(model_file, "wb").write(tflite_model)

  interpreter = tf.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

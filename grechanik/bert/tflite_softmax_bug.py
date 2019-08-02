import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  inp0 = tf.placeholder('float32', [10])
  res = tf.nn.softmax(inp0)
  res = tf.reshape(res, [1, 10])

  print(res.shape)

  #print(tf.get_default_graph().as_graph_def())

  def representative_dataset_gen():
    for i in range(10):
      yield [np.random.random_integers(0, 9, size=inp0.get_shape()).astype('float32'),]

  converter = tf.lite.TFLiteConverter.from_session(sess, [inp0], [res])
  converter.representative_dataset = representative_dataset_gen
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  tflite_model = converter.convert()
  model_file = "/tmp/tflite_softmax_bug.tflite"
  open(model_file, "wb").write(tflite_model)

  interpreter = tf.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

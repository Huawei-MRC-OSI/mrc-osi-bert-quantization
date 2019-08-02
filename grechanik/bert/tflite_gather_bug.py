import tensorflow as tf
import os
import numpy as np

with tf.Session() as sess:
  inp0 = tf.placeholder('int32', [4])
  embedding_table = tf.get_variable(name="emb",
                                    shape=[10, 16],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
  res = tf.gather(embedding_table, inp0)

  sess.run(tf.global_variables_initializer())

  print(res.shape)

  print(tf.get_default_graph().as_graph_def())

  def representative_dataset_gen():
    for i in range(10):
      yield [np.random.random_integers(0, 9, size=inp0.get_shape()).astype('int32')]

  converter = tf.lite.TFLiteConverter.from_session(sess, [inp0], [res])
  converter.representative_dataset = representative_dataset_gen
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  os.makedirs("/tmp/tflite_gather_bug_graphs", exist_ok=True)
  converter.dump_graphviz_dir = "/tmp/tflite_gather_bug_graphs"
  converter.dump_graphviz_video = True
  tflite_model = converter.convert()
  open("/tmp/tflite_gather_bug", "wb").write(tflite_model)

  interpreter = tf.lite.Interpreter(model_path="/tmp/tflite_gather_bug")
  interpreter.allocate_tensors()

  print("/tmp/tflite_gather_bug")

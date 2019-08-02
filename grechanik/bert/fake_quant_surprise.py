import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  inp1 = tf.placeholder('float32', [4, 16, 10])
  inp2 = tf.placeholder('float32', [4, 10, 20])
  res = tf.matmul(inp1, inp2)

  tf.io.write_graph(sess.graph_def, '/tmp', 'without_fake.pbtxt')

  tf.contrib.quantize.create_eval_graph()

  tf.io.write_graph(sess.graph_def, '/tmp', 'with_fake.pbtxt')

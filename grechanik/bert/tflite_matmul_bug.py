import tensorflow as tf

with tf.Session() as sess:
  inp1 = tf.placeholder('float32', [4, 48, 48])
  inp2 = tf.placeholder('float32', [4, 48, 64])
  res = tf.matmul(inp1, inp2)

  # Workaround
  #  muls = []
  #  for i in range(4):
  #    muls.append(tf.matmul(inp1[i, :, :], inp2[i, :, :]))
  #  res = tf.stack(muls)

  print(res.shape)

  #print(tf.get_default_graph().as_graph_def())

  converter = tf.lite.TFLiteConverter.from_session(sess, [inp1, inp2], [res])
  tflite_model = converter.convert()
  open("/tmp/converted_model.tflite", "wb").write(tflite_model)

  interpreter = tf.lite.Interpreter(model_path="/tmp/converted_model.tflite")
  interpreter.allocate_tensors()

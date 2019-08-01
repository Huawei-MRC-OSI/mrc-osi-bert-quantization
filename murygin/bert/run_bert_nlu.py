"""This is a bert classification script modified for the intent classification and slot filling.
It assumes that there is a python module called bert (can be installed with pip, or put your local
copy of pert into appropriate location).

export BERT_BASE_DIR="$HOME/proj/mrc-nlp/bert-google/uncased_L-12_H-768_A-12"

python run_bert_nlu.py \
  --do_train --do_eval \
  --data_dir=$HOME/proj/nlu-benchmark/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=48 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=$HOME/output/bert-nlu-$(date '+%Y-%m-%d_%H.%M.%S')

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import random
import json
import h5py
import my_bert_modeling as modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import numpy as np
import sklearn
from seqeval.metrics.sequence_labeling import f1_score as f1_score_seqeval

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "data_dir", os.path.expanduser("~/proj/nlu-benchmark/"),
    "The input data dir. Supposed to be the path to the nlu-benchamark repository.")

flags.DEFINE_string("task_name", 'nlu', "The name of the task to train.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "layer", -1,
    "The index of bert layer that is used as outputs.")

flags.DEFINE_float(
    "dev_set", 0.25,
    "How much of the train set to use as a dev set")

flags.DEFINE_bool(
    "eval_on_test", False,
    "Evaluate on the test set")

flags.DEFINE_bool(
    "profile_eval", False,
    "Enable profiling during evaluation")

flags.DEFINE_bool(
    "small_data", False,
    "Use this flag to prevent full data loading. Used for debugging")

flags.DEFINE_bool(
    "prune", False,
    "Perform pruning (work in progress, very slow and poor results)")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 200,
                     "How often to save summary.")

#  flags.DEFINE_integer("iterations_per_loop", 1000,
#                       "How many steps to make in each estimator call.")

#  flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

#  tf.flags.DEFINE_string(
#      "tpu_name", None,
#      "The Cloud TPU to use for training. This should be either the name "
#      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#      "url.")

#  tf.flags.DEFINE_string(
#      "tpu_zone", None,
#      "[Optional] GCE zone where the Cloud TPU is located in. If not "
#      "specified, we will attempt to automatically detect the GCE project from "
#      "metadata.")

#  tf.flags.DEFINE_string(
#      "gcp_project", None,
#      "[Optional] Project name for the Cloud TPU-enabled project. If not "
#      "specified, we will attempt to automatically detect the GCE project from "
#      "metadata.")

#  tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

#  flags.DEFINE_integer(
#      "num_tpu_cores", 8,
#      "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, *,
               input_tokenids, # [101, 123, 456, ...], ids of WordPiece tokens, padded and with cls
               input_mask,     # [1, 1, 1, ..., 1, 0, 0, 0, ...]
               input_segmentids, # Just zeros
               input_labelids, # [intent_id, slot_ids, ...]
               input_tokens, # ['[CLS]', 'a', '##b', ..., '[SEP]', 0, 0, ...]
               input_word_indices, # [-1, 0, 0, 1, 2, 2, ...], indices of words in wordinput
               wordinput_labelids, # label ids for word representation
               wordinput, # ['ab', 'the', 'one', ...], word representation
               is_real_example=True):
    self.input_tokenids = input_tokenids
    self.input_mask = input_mask
    self.input_segmentids = input_segmentids
    self.input_labelids = input_labelids
    self.input_tokens = input_tokens
    self.is_real_example = is_real_example
    self.input_word_indices = input_word_indices
    self.wordinput_labelids = wordinput_labelids
    self.wordinput = wordinput

  def untokenize_prediction(self, prediction):
    """Convert a list of labels for a fine-grained sentence representation into a list for
    the original coarser representation"""
    res = [0]*len(self.wordinput_labelids)
    for i, l in zip(self.input_word_indices, prediction):
      if i != -1:
        if res[i] == 0:
          res[i] = l
    return res


class DataProcessor(object):
  def __init__(self, data_dir):
    pass

  def get_train_examples(self):
    raise NotImplementedError()

  def get_dev_examples(self):
    raise NotImplementedError()

  def get_test_examples(self):
    raise NotImplementedError()

  def get_labels(self):
    raise NotImplementedError()

class NLUDataProcessor(DataProcessor):
  def __init__(self, data_dir, max_seq_length, tokenizer):
    if FLAGS.small_data:
      intents = ['AddToPlaylist']
    else:
      intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    self.intents = intents
    self.labels_to_ids = {'O': 0}
    self.labels = ['O']
    self.label_counts = [0]
    self.max_seq_length = max_seq_length

    self._add_labels(intents)

    train_data = []
    for intent in intents:
      f = open(data_dir + '/2017-06-custom-intent-engines/' + intent +
               '/train_' + intent + '_full.json', encoding='latin_1')
      sentences = json.load(f)[intent]
      f.close()
      for elem in sentences:
        sentence = [x['text'] for x in elem['data']]
        labels = [x['entity'] if 'entity' in x.keys() else 'O' for x in elem['data']]
        train_data.append(self.to_features(tokenizer, sentence, labels, intent))

    test_data = []
    for intent in intents:
      f = open(data_dir + '/2017-06-custom-intent-engines/' + intent +
               '/validate_' + intent + '.json', encoding='latin_1')
      sentences = json.load(f)[intent]
      f.close()
      for elem in sentences:
        sentence = [x['text'] for x in elem['data']]
        labels = [x['entity'] if 'entity' in x.keys() else 'O' for x in elem['data']]
        test_data.append(self.to_features(tokenizer, sentence, labels, intent, is_test=True))

    random.shuffle(test_data)
    self.test_data = test_data
    random.shuffle(train_data)
    self.train_data = train_data

  def to_features(self, tokenizer, data_pieces, label_pieces, intent, is_test=False):
    wordinput = []
    wordinput_labels = []
    input_tokens = ['[CLS]']
    token_labels = [intent]
    input_word_indices = [-1]
    word_index = 0
    for piece, label in zip(data_pieces, label_pieces):
      words = piece.split()
      if label == 'O':
        word_labels = [label]*len(words)
      else:
        word_labels = ["B-" + label] + ["I-" + label]*(len(words) - 1)
      wordinput += words
      wordinput_labels += word_labels

      for d, l in zip(words, word_labels):
        d_tokens = tokenizer.tokenize(d)
        input_tokens += d_tokens
        input_word_indices += [word_index]*len(d_tokens)
        word_index += 1
        token_labels += [l]*len(d_tokens)

    self._add_labels(token_labels)

    if len(input_tokens) >= self.max_seq_length:
      print("Max length exceeded: ", len(input_tokens))

    input_tokens = input_tokens[:self.max_seq_length-1]
    token_labels = token_labels[:self.max_seq_length-1]

    for l in token_labels:
      self.label_counts[self.labels_to_ids[l]] += 1

    input_tokens += ['[SEP]']
    token_labels += ['O']
    input_word_indices += [-1]

    input_tokenids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_labels = [self.labels_to_ids[l] for l in token_labels]
    input_mask = [1] * len(input_tokens)

    # Zero-pad up to the sequence length.
    while len(input_tokenids) < self.max_seq_length:
      input_tokenids.append(0)
      input_mask.append(0)
      token_labels.append(0)
      input_word_indices.append(-1)

    wordinput_labelids = [self.labels_to_ids[l] for l in wordinput_labels]

    return InputFeatures(input_tokenids=input_tokenids,
                         input_mask=input_mask,
                         input_segmentids=[0]*self.max_seq_length,
                         input_labelids=token_labels,
                         input_tokens=input_tokens,
                         input_word_indices=input_word_indices,
                         wordinput_labelids=wordinput_labelids,
                         wordinput=wordinput)

  def _add_labels(self, labels):
    for l in labels:
      if l not in self.labels_to_ids:
        self.labels_to_ids[l] = len(self.labels_to_ids)
        self.labels.append(l)
        self.label_counts.append(0)

  def get_train_examples(self):
    return self.train_data[:int(len(self.train_data)*(1 - FLAGS.dev_set))]

  def get_dev_examples(self):
    return self.train_data[int(len(self.train_data)*(1 - FLAGS.dev_set)):]

  def get_test_examples(self):
    return self.test_data

  def get_labels(self):
    return self.labels


def examples_to_file(examples, output_file):
  """Convert a set of InputFeatures to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_tokenids"] = create_int_feature(example.input_tokenids)
    features["input_mask"] = create_int_feature(example.input_mask)
    features["input_segmentids"] = create_int_feature(example.input_segmentids)
    features["input_labelids"] = create_int_feature(example.input_labelids)
    features["is_real_example"] = create_int_feature(
        [int(example.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_tokenids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "input_segmentids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_labelids": tf.FixedLenFeature([seq_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_tokenids, input_mask, input_segmentids,
                 labels, num_labels, use_one_hot_embeddings, label_counts):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_tokenids,
      input_mask=input_mask,
      token_type_ids=input_segmentids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_all_encoder_layers()[FLAGS.layer]

  hidden_size = output_layer.shape[-1].value

  output_weights = \
    tf.get_variable("output_weights",
                    [hidden_size, num_labels],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = \
    tf.get_variable("output_bias",
                    [num_labels],
                    initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.einsum("ijk,kl->ijl", output_layer, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    #  label_counts_sum = sum(label_counts)
    #  label_weights = [1.0/c for c in label_counts]
    #  label_weights = tf.constant([20*c/sum(label_weights) for c in label_weights], dtype=tf.float32)
    #  label_weights = tf.gather(label_weights, labels)

    masked_weights = tf.cast(input_mask, dtype='float32')
    weights = tf.expand_dims(masked_weights, axis=-1)
    per_example_loss = -tf.reduce_sum(weights * one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities, model)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, label_counts, prune_hook=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_tokenids = features["input_tokenids"]
    input_mask = features["input_mask"]
    input_segmentids = features["input_segmentids"]
    input_labelids = features["input_labelids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(input_labelids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities, model) = create_model(
        bert_config, is_training, input_tokenids, input_mask, input_segmentids, input_labelids,
        num_labels, use_one_hot_embeddings, label_counts=label_counts)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    #  internal_values = {}
    #  for name, value in model.internal_values.items():
    #    value_shape = tf.shape(value)
    #    batch_size = tf.shape(probabilities)[0]
    #    new_two_dims = tf.convert_to_tensor([batch_size, value_shape[0]//batch_size], dtype='int32')
    #    new_shape = tf.concat([new_two_dims, value_shape[1:]], 0)
    #    value = tf.reshape(value, new_shape)
    #    indices = tf.cast(tf.random.uniform([batch_size, 1])*tf.cast(new_shape[1], 'float32'),
    #                      'int32')
    #    internal_values[name] = tf.batch_gather(value, indices)
    #    print(name, internal_values[name].shape)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      if prune_hook is not None:
        prune_hook.model = model
      
      # Quantization-aware training graph transformation
      tf.contrib.quantize.create_training_graph(tf.get_default_graph(), 5000)

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      
      # Quantization-aware eval graph transformation
      tf.contrib.quantize.create_eval_graph(tf.get_default_graph())

      predicted_labels = tf.squeeze(tf.argmax(logits, axis=-1, output_type=tf.int32))
      slot_filling_accuracy = tf.metrics.accuracy(input_labelids[:, 1:], predicted_labels[:, 1:],
                                                  weights=input_mask[:, 1:])
      intent_prediction_accuracy = tf.metrics.accuracy(input_labelids[:, :1], predicted_labels[:, :1])

      # Whole Frame Accuracy: this is the percentage of propositions for which
      # there is an exact match between the proposed and correct labelings.
      whether_equal = tf.logical_or(tf.logical_not(tf.cast(input_mask, dtype=tf.bool)),
                                    tf.equal(input_labelids, predicted_labels))
      whole_frame_accuracy = \
        tf.metrics.mean(tf.cast(tf.reduce_all(whether_equal, axis=-1), 'float32'))

      total_mask_bits = 0
      unmasked = 0
      for mask in model.sparsity_masks.values():
        total_mask_bits = tf.shape(mask, out_type='float32')[-1] + total_mask_bits
        unmasked = tf.reduce_sum(mask) + unmasked

      eval_metrics = {
          "slot_filling_accuracy": slot_filling_accuracy,
          "intent_prediction_accuracy": intent_prediction_accuracy,
          "whole_frame_accuracy": whole_frame_accuracy,
          "unmasked_ratio": (unmasked/total_mask_bits, tf.no_op())
      }

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics,
          predictions={"probabilities": probabilities})
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities})
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_tokenids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.input_segmentids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_tokenids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_segmentids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_labelids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

class PruneHook(tf.train.SessionRunHook):
  def __init__(self, every_n_steps=50, drop_num=1000):
    self.every_n_steps = every_n_steps
    self.drop_num = drop_num
    self.cur_step = None
    self.model = None

  def begin(self):
    self.cur_step = 0

  def after_create_session(self, session, coord):
    self.cur_step = 0

  def before_run(self, run_context):
    if self.cur_step % self.every_n_steps == 0:
      pass

  def after_run(self, run_context, run_values):
    if self.cur_step % self.every_n_steps == 0:
      print("pruning")
      tf.get_default_graph()._unsafe_unfinalize()
      with run_context.session.graph.as_default():
        lst = list(self.model.sparsity_masks.values())
        assignments = []
        print("building graph")
        for i in range(self.drop_num):
          n = random.randrange(0, len(self.model.sparsity_masks))
          mask = lst[n]
          k = random.randrange(0, mask.shape[-1])
          assignments.append(mask[k].assign(0.0))
        print("running assignments")
        run_context.session.run(assignments)
      print("assigned")
    self.cur_step += 1


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  random.seed(42)

  processors = {
      "nlu": NLUDataProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s, known_tasks are %s" % (task_name, processors.keys()))

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  processor = processors[task_name](FLAGS.data_dir, FLAGS.max_seq_length, tokenizer)

  label_list = processor.get_labels()

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  #session_config.gpu_options.per_process_gpu_memory_fraction=0.2

  # Specify output directory and number of checkpoint steps to save
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_summary_steps=FLAGS.save_summary_steps,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=session_config)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples()
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  prune_hook = PruneHook()

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False,
      label_counts=processor.label_counts,
      prune_hook=prune_hook)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": FLAGS.train_batch_size})

  if FLAGS.eval_on_test:
    eval_examples = processor.get_test_examples()
  else:
    eval_examples = processor.get_dev_examples()

  num_actual_eval_examples = len(eval_examples)

  eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
  examples_to_file(eval_examples, eval_file)

  # This tells the estimator to run through the entire set.
  eval_drop_remainder = False
  eval_input_fn = file_based_input_fn_builder(
      input_file=eval_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=eval_drop_remainder)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    examples_to_file(train_examples, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                        hooks=([prune_hook] if FLAGS.prune else []))
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      start_delay_secs=60, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    if FLAGS.profile_eval:
      with tf.contrib.tfprof.ProfileContext(FLAGS.output_dir) as pctx:
        eval_result = estimator.evaluate(input_fn=eval_input_fn)
    else:
      eval_result = estimator.evaluate(input_fn=eval_input_fn)

    pred_result = estimator.predict(input_fn=eval_input_fn)

    # Slot ids from all examples
    all_label_ids = []
    all_pred_ids = []

    # Slot labels for all sentences
    seqeval_true_labels = []
    seqeval_pred_labels = []

    # Internal values
    internal_values = {}

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      mispredictions_count = 0
      for example, prediction in zip(eval_examples, pred_result):
        probabilities = prediction["probabilities"]
        full_pred_ids = np.argmax(probabilities, axis=-1)
        full_true_ids = example.input_labelids
        wordinput_pred_ids = example.untokenize_prediction(full_pred_ids)
        wordinput_true_ids = example.wordinput_labelids
        true_label_names = [processor.labels[l] for l in wordinput_true_ids]
        pred_label_names = [processor.labels[l] for l in wordinput_pred_ids]
        full_true_label_names = [processor.labels[l] for l in full_true_ids]
        full_pred_label_names = [processor.labels[l] for l in full_pred_ids]

        all_label_ids += list(wordinput_true_ids)[1:]
        all_pred_ids += list(wordinput_pred_ids)[1:]

        seqeval_true_labels.append(true_label_names)
        seqeval_pred_labels.append(pred_label_names)

        writer.write("\n")
        writer.write("%s\n" % " ".join(example.input_tokens))
        writer.write("%s\n" % str(wordinput_true_ids))
        writer.write("%s\n" % list(zip(example.input_tokens, full_true_label_names)))
        writer.write("%s\n" % list(zip(example.input_tokens, full_pred_label_names)))
        writer.write("%s\n" % list(zip(example.wordinput, true_label_names)))
        writer.write("%s\n" % list(zip(example.wordinput, pred_label_names)))
        for token, correct, predicted in zip(example.input_tokens, full_true_ids, full_pred_ids):
          if correct != predicted:
            writer.write("ERR: %s is %s but predicted %s\n" %
                         (token, processor.labels[correct], processor.labels[predicted]))

        if (np.array(wordinput_pred_ids) != np.array(wordinput_true_ids)).any() or \
            full_pred_ids[0] != full_true_ids[0]:
          mispredictions_count += 1
          writer.write("CONSIDERED WRONG!\n")
        else:
          writer.write("CONSIDERED RIGHT!\n")

        #  for name, value in prediction.items():
        #    if name != 'probabilities':
        #      internal_values.setdefault(name, []).append(np.expand_dims(value, 0))

      for average in ['micro', 'macro', 'weighted']:
        eval_result['f1_' + average] = \
          sklearn.metrics.f1_score(all_label_ids, all_pred_ids, average=average)

      eval_result['f1_score_seqeval'] = f1_score_seqeval(seqeval_true_labels, seqeval_pred_labels)

      eval_result['recomputed_wfa'] = 1 - mispredictions_count / num_actual_eval_examples
      eval_result['recomputed_slot_acc'] = \
        sklearn.metrics.accuracy_score(all_label_ids, all_pred_ids)

      tf.logging.info("***** Eval results *****")
      for key in sorted(eval_result.keys()):
        tf.logging.info("  %s = %s", key, str(eval_result[key]))
        writer.write("%s = %s\n" % (key, str(eval_result[key])))

    tf.logging.info("***** Eval results *****")
    for key in sorted(eval_result.keys()):
      tf.logging.info("  %s = %s", key, str(eval_result[key]))

    #  with h5py.File(os.path.join(FLAGS.output_dir, "internal_values.h5"), 'w') as internal_values_h5:
    #    for name in internal_values:
    #      print("Writing", name)
    #      internal_values_h5.create_dataset(name, data=np.concatenate(internal_values[name]))

    print(output_eval_file)


  # Actions for serving model in savedmodel format
  def serving_input_receiver_fn():
    features = {
      "input_tokenids": tf.placeholder(shape=[1, FLAGS.max_seq_length], dtype=tf.int32),
      "input_mask": tf.placeholder(shape=[1, FLAGS.max_seq_length], dtype=tf.int32),
      "input_segmentids": tf.placeholder(shape=[1, FLAGS.max_seq_length], dtype=tf.int32),
      "input_labelids": tf.placeholder(shape=[1, FLAGS.max_seq_length], dtype=tf.int32),
      "is_real_example": tf.placeholder(shape=[1], dtype=tf.int32)
    }
    return tf.estimator.export.ServingInputReceiver(features, features)

  estimator.export_saved_model("/workspace/bert/savedmodel", serving_input_receiver_fn, as_text=True)

if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

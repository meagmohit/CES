from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.ops import rnn, rnn_cell

sys.path.append(os.getcwd()) 

import input_fast as input


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summaries_train_dir', 'logs/summaries_train/', """Where to save summary logs for training in TensorBoard.""")
tf.app.flags.DEFINE_string('summaries_finetune_dir', 'logs/summaries_finetune/', """Where to save summary logs for finetuning in TensorBoard.""")
tf.app.flags.DEFINE_string('model_train_dir','logs/models_trained',""" Checkpoints and trained models to finetune them""")
tf.app.flags.DEFINE_string('model_finetune_dir','logs/models_finetuned', """ Checkpoints and finetuned models saved""")

tf.app.flags.DEFINE_string('mode','',""" possible: train, finetune""")
tf.app.flags.DEFINE_string('eval_mode','', """ train/finetune : use corresponding model weights to evaluate on test dataset  """)

tf.app.flags.DEFINE_integer('iterations',10000, """ Total number of training iterations (not epochs) """)
tf.app.flags.DEFINE_string('data_path','data',""" Path of all the Video data and frames""")
tf.app.flags.DEFINE_integer('saving_iter',1000,"""  Iterations after which model file should be updated  """)
tf.app.flags.DEFINE_float('learning_rate_decay',0.8, """ Iterations after which learning rate will be decayed by the given factor  """)
tf.app.flags.DEFINE_integer('learning_rate_iter',1000,""" The factor with which learning rate will be decayed  """)


class Config(object):
  #Parameter which can't be changed once the model is created
  hidden_size_lstm = 16
  hidden_size_fc1 = 4
  num_lstm_layers = 2
  num_class = 2
  learning_rate = 0.1
  #Parameters which can be changed after model instantiation
  batch_size = 16
  step_size = 256  # 256Hz, and 1s of data, -0.2 to 0.8ms or 0 to 1 ms
  num_electrodes = 25
  features = (9,11,14) # Pz,Fz and Cz

class model(object):

  def __init__(self, is_training, config):
   
    self._batch_size = batch_size = config.batch_size 
    self._num_electrodes = num_electrodes = config.num_electrodes
    self._step_size = step_size = config.step_size
    self._hidden_size_lstm = hidden_size_lstm = config.hidden_size_lstm
    self._hidden_size_fc1 = hidden_size_fc1 = config.hidden_size_fc1
    self._num_lstm_layers = num_lstm_layers = config.num_lstm_layers   
    self._num_class = num_class = config.num_class
    self._num_features = num_features = len(config.features)
    self._learning_rate = tf.Variable(config.learning_rate, trainable=False)

    with tf.name_scope('input'):
      self._input_feature = input_feature = tf.placeholder(tf.float32,  shape=[batch_size, step_size, num_features], name='feature_input')
      self._input_targets = input_targets = tf.placeholder(tf.int32, shape=[batch_size, num_class])

    layer_name = 'pre-LSTM'
    with tf.name_scope('layer_name'):
      _X = tf.transpose(input_feature, [1, 0, 2])
      _X = tf.reshape(_X, [-1, num_features])
      pl_weights_1 = tf.Variable(tf.truncated_normal([num_features, hidden_size_lstm], stddev=0.001), name='pl_weights_1')
      pl_biases_1 = tf.Variable(tf.zeros([hidden_size_lstm]), name='pl_biases_1')
      _X = tf.nn.relu(tf.matmul(_X, pl_weights_1) + pl_biases_1)
      _X = tf.split(0, step_size, _X)

    layer_name = 'LSTM'
    with tf.name_scope(layer_name):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layers)
    
      outputs, state = rnn.rnn(cell, _X, dtype=tf.float32)
      self._state = state
      #output = tf.reshape(tf.concat(1,outputs), [-1, hidden_size_lstm])
      output = outputs[-1]

    layer_name = 'fully_connected_1'
    with tf.name_scope(layer_name):
      with tf.name_scope('weights_1'):
        self._fc_weights_1 = fc_weights_1 = tf.Variable(tf.truncated_normal([hidden_size_lstm, hidden_size_fc1], stddev=0.001), name='fc_weights_1')
        self.variable_summaries(fc_weights_1, layer_name + '/weights_1')
      with tf.name_scope('biases_1'):
        self._fc_biases_1 = fc_biases_1 = tf.Variable(tf.zeros([hidden_size_fc1]), name='fc_biases_1')
        self.variable_summaries(fc_biases_1, layer_name + '/biases_1')
      with tf.name_scope('Wx_plus_b_1'):
        self._final_logits_1 = final_logits_1 = tf.matmul(output, fc_weights_1) + fc_biases_1
        tf.summary.histogram(layer_name + '/pre_activations_final_1', final_logits_1)

    layer_name = 'fully_connected'
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        self._fc_weights = fc_weights = tf.Variable(tf.truncated_normal([hidden_size_fc1, num_class], stddev=0.001), name='fc_weights')
        self.variable_summaries(fc_weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        self._fc_biases = fc_biases = tf.Variable(tf.zeros([num_class]), name='fc_biases')
        self.variable_summaries(fc_biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        self._final_logits = final_logits = tf.matmul(final_logits_1, fc_weights) + fc_biases
        tf.summary.histogram(layer_name + '/pre_activations_final', final_logits)

    self._final_tensor = final_tensor = tf.nn.softmax(final_logits, name='final_result')
    tf.summary.histogram('final_result' + '/activations', final_tensor)


    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.cast(tf.equal(tf.argmax(final_logits, 1), tf.argmax(self._input_targets, 1)),tf.float32)
      with tf.name_scope('accuracy'):
        self._evaluation_step = evaluation_step = tf.reduce_mean(correct_prediction)
      tf.summary.scalar('accuracy', evaluation_step)

   
    if not is_training:
      return

    # Part exclusively required for training the model 
    with tf.name_scope('loss'):
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._input_targets, logits=final_logits)
      with tf.name_scope('total'):
        self._loss_mean = loss_mean = tf.reduce_mean(loss)
      tf.summary.scalar('loss',loss_mean)

    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(self._learning_rate)
      self._train_step = train_step = optimizer.minimize(loss_mean)

    self._new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._learning_rate_update = tf.assign(self._learning_rate, self._new_learning_rate)    

  def assign_learning_rate(self, session, learning_rate_value):
    session.run(self._learning_rate_update, feed_dict={self._new_learning_rate: learning_rate_value})

  def variable_summaries(self, var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.summary.scalar('sttdev/' + name, stddev)
      tf.summary.scalar('max/' + name, tf.reduce_max(var))
      tf.summary.scalar('min/' + name, tf.reduce_min(var))
      tf.summary.histogram(name, var)

  @property
  def input_feature(self):
    return self._input_feature

  @property
  def input_targets(self):
    return self._input_targets

  @property
  def sequence_len(self):
    return self._sequence_len

  @property
  def final_tensor(self):
    return self._final_tensor

  @property
  def loss_mean(self):
    return self._loss_mean

  @property
  def train_step(self):
    return self._train_step

  @property
  def evaluation_step(self):
    return self._evaluation_step

  @property
  def state(self):
    return self._state

  @property
  def correct_predictions(self):
    return self._correct_prediction

  @property
  def learning_rate(self):
    return self._learning_rate
  
  #@property
  #def grads_and_vars(self):
  #  return self._grads_and_vars

  #@property
  #def tvars(self):
  #  return self._tvars

  #@property
  #def grads(self):
  #  return self._grads

  #@property
  #def final_result(self):
  #  return self._final_result

def _evaluation_operation():
  if FLAGS.eval_mode == 'train':
    model_dir = FLAGS.model_train_dir
  elif FLAGS.eval_mode == 'finetune':
    model_dir = FLAGS.model_finetune_dir
  else:
    print("\033[91m eval_mode not selected, chosse --eval_mode= train or finetune \033[0m ")
    return

  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
  sess = tf.Session()
  config = Config()
  config.batch_size = 1
  print("\033[94m Loading the model... \033[0m")
  m = model(is_training=False, config=config)
  print("\033[94m Preparing the input stream for testing... \033[0m")
  input_stream = input.input(FLAGS.data_path, config, image_data_tensor=jpeg_data_tensor, bottleneck_tensor=bottleneck_tensor)
  
  saver = tf.train.Saver(tf.all_variables())
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("\033[94m Loading the pre-trained weights of model... \033[0m")
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("\033[91m Opted for evaluation mode, but no pre-trained/pre-finetuned model found \033[0m")
    return 
  
  dacc_frame = np.zeros([config.num_class, config.num_class])
  dacc_cumul = np.zeros([config.num_class,config.num_class])
  seqlen_frame = np.zeros([config.num_class])
  seqlen_cumul = np.zeros([config.num_class])

  total_videos = input_stream.test_total
  for i in range(input_stream.test_total):
    print(" \033[94m Currently on : %d / %d \033[0m " %(i,total_videos))
    input_feature, sequence_len, input_targets = input_stream.next_batch(sess, flag='eval')
    feed_dict = {m.input_feature: input_feature, m.input_targets: input_targets, m.sequence_len: sequence_len}
    final_result  = sess.run([m.final_result], feed_dict=feed_dict)

    label = input_targets[0,0]
    temp_list = np.zeros([config.num_class])
    for j in range(sequence_len[0]):
      temp_list[final_result[0][j]] += 1

    dacc_frame[label,:] += temp_list
    seqlen_frame[label] += sequence_len[0]
    pool_result = np.argmax(temp_list)
    dacc_cumul[label,pool_result] += 1
    seqlen_cumul[label] += 1

  acc_frame = dacc_frame.trace()/sum(seqlen_frame)
  acc_cumul = dacc_cumul.trace()/sum(seqlen_cumul)
  print("\033[93m Accuracy:   Frame = %f  Cumulative = %f \033[0m " %(acc_frame, acc_cumul))
  for i in range(config.num_class):
    dacc_frame[i,:] /= seqlen_frame[i]
    dacc_cumul[i,:] /= seqlen_cumul[i]
  np.savetxt("delta_acc_frame.csv", np.asarray(dacc_frame), delimiter=",")
  np.savetxt("delta_acc_cumul.csv", np.asarray(dacc_cumul), delimiter=",")
    


# Starting the main function
def main(_):

  # Choose between modes - train/finetune/evaluate
  if FLAGS.mode == 'train':
    summaries_dir = FLAGS.summaries_train_dir
    model_dir = FLAGS.model_train_dir
    print("\033[95m Mode opted: Training from scratch... \033[0m")
  elif FLAGS.mode == 'finetune':
    summaries_dir = FLAGS.summaries_finetune_dir
    model_dir = FLAGS.model_finetune_dir
    print("\033[95m Mode opted: Finetuning the pre-trained model ... \033[0m")
  elif FLAGS.mode == 'evaluation':
    print("\033[95m Mode opted: Evaluating the model on test data set ... \033[0m")
    _evaluation_operation()
    return
  else:
    print(" \033[91m Provide Either train or finetune.... Other Implementations pending//// \033[0m")
    return

  #Clearing summaries and model directories if exists
  if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)
  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)
  tf.gfile.MakeDirs(model_dir)

  

  checkpoint_path = os.path.join(model_dir, 'model.ckpt')
  
  sess = tf.Session()
  config = Config()
  print("\033[94m Loading the model... \033[0m") 
  m = model(is_training=True, config=config)  
  print("\033[94m Preparing the input stream for training... \033[0m")
  input_stream = input.input(FLAGS.data_path, config)
  print("\033[94m Initializing the network... \033[0m")
  merged = tf.summary.merge_all()
  saver = tf.train.Saver(tf.global_variables())
  train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
  test_writer = tf.summary.FileWriter(summaries_dir+'/test',sess.graph)
  init = tf.global_variables_initializer()
  sess.run(init)

  global_step = 0
  if FLAGS.mode == 'finetune':
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_train_dir) 
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      print("\033[91m Opted for finetune mode, but no pre-trained model found \033[0m") 
      return

  print("\033[94m Training begins... \033[0m")
  #state = sess.run(m.initial_state)
  for i in range(FLAGS.iterations):
   
    input_feature, input_targets = input_stream.next_batch(sess, flag='train')
    feed_dict = {m.input_feature: input_feature, m.input_targets: input_targets}
    train_summary, _ = sess.run([merged, m.train_step], feed_dict=feed_dict)
    train_writer.add_summary(train_summary, i)
    train_accuracy, loss_value, learning_rate = sess.run([m.evaluation_step, m.loss_mean, m.learning_rate], feed_dict=feed_dict)
    
    input_feature, input_targets = input_stream.next_batch(sess, flag='test')
    feed_dict = {m.input_feature: input_feature, m.input_targets: input_targets}
    test_summary, test_accuracy = sess.run([merged, m.evaluation_step], feed_dict=feed_dict)
    test_writer.add_summary(test_summary, i)


    print('Step %d:  Learning Rate = %f  Train Loss = %f  Train Accuracy = %f  Test_Accuracy = %f' % (i, learning_rate, loss_value, train_accuracy, test_accuracy))
    if (i % FLAGS.saving_iter) == 0:
      saver.save(sess, checkpoint_path, global_step=i+global_step) 
    if (((i % FLAGS.learning_rate_iter) == 0) & (i != 0)):
      m.assign_learning_rate(sess, FLAGS.learning_rate_decay*learning_rate)


if __name__ == '__main__':
  tf.app.run()

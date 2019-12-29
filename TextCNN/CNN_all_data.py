#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import pandas as pd
import os

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1566023583/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

def evalate(x):
    x_test = np.array(list(vocab_processor.transform(x)))
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    return all_predictions

SPATH = 'API_SO_Sentences/'
TPATH = 'API_SO_Titles/'

file_list = os.listdir(SPATH)
for index in range(0, len(file_list)):
    file = file_list[index]
    apiname = file[:-10]
    print('apiname:', apiname)
    ID = []
    x_raw = []
    y_test = []
    df = pd.DataFrame(pd.read_csv(SPATH+file, encoding='ISO-8859-1', header=0))
    print(df.shape[0])
    for index in range(0, df.shape[0]):
        if index%1000 == 0:
            print(index)
        ID.append(df.iloc[index].Id)
        x_raw.append(str(df.iloc[index].Sentence))
    all_predictions = evalate(x_raw)
    towrite = {'Id':ID, 'Sentence':x_raw, 'Label':all_predictions}
    dataframe = pd.DataFrame(towrite)
    dataframe.to_csv('all_sent_results/'+apiname+'_sentsp.csv',index=False)

file_list = os.listdir(TPATH)
for index in range(0, len(file_list)):
    file = file_list[index]
    apiname = file[:-10]
    print('apiname:', apiname)
    ID = []
    x_raw = []
    y_test = []
    df = pd.DataFrame(pd.read_csv(TPATH+file, encoding='ISO-8859-1', header=0))
    print(df.shape[0])
    for index in range(0, df.shape[0]):
        if index%1000 == 0:
            print(index)
        ID.append(df.iloc[index].Id)
        x_raw.append(str(df.iloc[index].Title))
    all_predictions = evalate(x_raw)
    towrite = {'Id':ID, 'Title':x_raw, 'Label':all_predictions}
    dataframe = pd.DataFrame(towrite)
    dataframe.to_csv('all_title_results/'+apiname+'_titlep.csv',index=False)

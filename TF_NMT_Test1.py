import sys
sys.version
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import matplotlib.backends.backend_pdf
import datetime

import tensorflow as tf


import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from puma import DCA_DL,helpers

# sys.path.append("X:/SCRIPTS/Python ML/TEST")
# from DCA_DL import *
# from helpers import *
# from pycanaan_basic_function import *


### Training Data type either "SIMPLE" or "COMPLEX" => COMPLEX version includes different segmented versions (only Flat to HYP2EXP for now) and EXP and etc. Will add additional types later
train_data_type="COMPLEX"
number_of_samples = 5
if train_data_type.upper()=="SIMPLE":
    ### Generate Input Parameters
    input_ls=DCA_DL.generate_dca_parm_ls(no_samples=number_of_samples,seedno=2222,noisy_data_pct=0.8,terminal_de=0.06,seq_length=60,mix_pct=True)
    test_input_ls=DCA_DL.generate_dca_parm_ls(no_samples=number_of_samples,seedno=2024,noisy_data_pct=0.2,terminal_de=0.06,seq_length=60,mix_pct=True)

    ### Generate Raw Production Stream to train TF
    if __name__=='__main__':
        mp_engine = Pool()
        cmb_result_orig = mp_engine.map(DCA_DL.generate_tensorflow_proddata_orig, input_ls)
        cmb_result_noisy= mp_engine.map(DCA_DL.enerate_tensorflow_proddata_noisy, input_ls)
        test_orig = mp_engine.map(DCA_DL.generate_tensorflow_proddata_orig, test_input_ls)
        test_noisy = mp_engine.map(DCA_DL.generate_tensorflow_proddata_noisy, test_input_ls)


    x_train=np.asarray(cmb_result_noisy)
    y_train=np.asarray(cmb_result_orig)
    raw_seq_ls=[xx[5] for xx in input_ls]
    seq_len_train=np.asarray(raw_seq_ls)

    x_plot_test=np.asarray(test_noisy)
    y_plot_test=np.asarray(test_orig)

else:
    ### Generate Input Parameters for Training and Testing / Verification
    input_ls = DCA_DL.generate_dca_parm_ls_variousprod(no_samples=number_of_samples, seedno=21231, noisy_data_pct=0.8, terminal_de=0.06,
                                                seq_length=60, min_length=3)
    test_input_ls = DCA_DL.generate_dca_parm_ls_variousprod(no_samples=number_of_samples, seedno=1211, noisy_data_pct=0.8, terminal_de=0.06,
                                                     seq_length=60, min_length=3)

    ### Generate Raw Production vectors (Using Parallel Processing Pool CPU) - Generate Clean Production Vector and Noise added Production Vector

    if __name__ == '__main__':
        mp_engine = Pool()
        # Generating Training Data
        cmb_result_orig = mp_engine.map(DCA_DL.generate_tensorflow_proddata_variousprod_sl, input_ls)
        cmb_result_noisy = mp_engine.map(DCA_DL.generate_tensorflow_proddata_variousprod_noisfy_sl, input_ls)

      # Generating Testing / Verification Data
        test_orig = mp_engine.map(DCA_DL.generate_tensorflow_proddata_variousprod_sl, test_input_ls)
        test_noisy = mp_engine.map(DCA_DL.generate_tensorflow_proddata_variousprod_noisfy_sl, test_input_ls)

    x_train = np.asarray(cmb_result_noisy)
    y_train = np.asarray(cmb_result_orig)
    raw_seq_ls = [xx[5] for xx in input_ls]
    seq_len_train = np.asarray(raw_seq_ls)

    x_plot_test = np.asarray(test_noisy)
    y_plot_test = np.asarray(test_orig)

### Use GPU or CPU Flag
which_device_touse="GPU"
# Model Save location
save_loc="X:/SCRIPTS/Python ML/TEST/tf_model"

tf_board_save_loc="X:/SCRIPTS/Python ML/TEST/tf_model/tfboard"


if which_device_touse.upper() =="GPU":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.70
else:
    ### Testintg on CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

train_type="NEW"  ### NEW / UPDATE




### Helper Dynamic unrolling

xt, xlen = helpers.batch(cmb_result_orig)

sequence_lengths = [len(seq) for seq in cmb_result_orig]
batch_size = len(inputs)

if max_sequence_length is None:
    max_sequence_length = max(sequence_lengths)



cmb_result_orig


helpers.batch

tf.contrib.seq2seq.Helper



len(cmb_result_orig)

















### Start DL Process
tf.reset_default_graph()  # We didn't have any previous graph objects running, but this would reset the graphs

# Run Settings
learning_rate = 0.00001  # small learning rate so we don't overshoot the minimum (change x3)
epochs = 2000  # number of iterations or training cycles, includes both the FeedFoward and Backpropogation




num_periods = 1  # number of periods per vector we are using to predict one period ahead
inputs = 60  # number of vectors submitted
hidden = 800  # number of neurons we will recursively work through, can be changed to improve accuracy
output = 60  # number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])  # create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output])

### Multi Layer Model
def multilayer_perceptron(X,weights,biases):
    # cv_X= tf.unstack(X,[-1,inputs])
    cv_X = tf.reshape(X,[-1,inputs])
    # Hidden Layer with RELU Activation
    #  X x Weights + Biases
    layer_1 = tf.add(tf.matmul(cv_X,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden Layer with RELU Activation
    layer_2 = tf.add( tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output Layer with Linear Activation
    out_layer = tf.add(tf.matmul(layer_2,weights['out']),biases['out'])
    # out_layer = tf.nn.relu(out_layer)

    return out_layer


## Store Weights and Biases
weights = {
    'h1': tf.Variable(tf.random_normal([inputs,hidden])),
    'h2': tf.Variable(tf.random_normal([hidden,hidden])),
    'out': tf.Variable(tf.random_normal([hidden,output]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden])),
    'b2': tf.Variable(tf.random_normal([hidden])),
    'out': tf.Variable(tf.random_normal([output]))
}

### Model
predictor = multilayer_perceptron(X,weights,biases)







###############################!!!!!!!!!!!!!!!!!!!!!!!!!#########################
def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.unstack(X, output, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, X, dtype=tf.float32,
                                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']























# Loss  & Optimizer
with tf.variable_scope("loss_fn"):
    loss = tf.reduce_sum(tf.square(predictor - y))  # define the cost function which evaluates the quality of our model
    tf.summary.scalar("loss_value",loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # gradient descent method
    training_op = optimizer.minimize(
        loss)  # train the result of the application of the cost_function



#
#
# with tf.variable_scope("rnn_output"):
#     stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])  # change the form into a tensor
#     stacked_outputs = tf.layers.dense(stacked_rnn_output, output)  # specify the type of layer (dense)
#     outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])  # shape of results



merged_summary_op = tf.summary.merge_all()



init = tf.global_variables_initializer()  # initialize all the variables


if train_type.upper()=="NEW":

    with tf.Session(config=config) as sess:
        ts_saver = tf.train.Saver()
        tf_board_writer=tf.summary.FileWriter(tf_board_save_loc,sess.graph)
        init.run()
        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_train, y: y_train})
            summary_str = sess.run(merged_summary_op, feed_dict={X: x_train, y: y_train})
            tf_board_writer.add_summary(summary_str,ep)
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_train, y: y_train})
                print(ep, "\tMSE:", mse)

        # test_y_pred=sess.run(predictor,feed_dict={X: x_plot_test})
        save_path = ts_saver.save(sess, save_loc+"/garfield.ckpt")
        print("Model saved in file: %s" % save_path)

else:
    with tf.Session(config=config) as sess:
        ts_saver = tf.train.Saver()

        ts_saver.restore(sess,save_loc+"/garfield.ckpt")
        tf_board_writer = tf.summary.FileWriter(tf_board_save_loc, sess.graph)

        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_train, y: y_train})
            summary_str = sess.run(merged_summary_op, feed_dict={X: x_train, y: y_train})
            # summary_str=sess.run(training_op, feed_dict={X: x_train, y: y_train})
            tf_board_writer.add_summary(summary_str, ep)
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_train, y: y_train})
                print(ep, "\tMSE:", mse)

        test_y_pred=sess.run(outputs,feed_dict={X: x_plot_test})
        save_path = ts_saver.save(sess, save_loc+"/garfield.ckpt")
        print("Model saved in file: %s" % save_path)






### Save case for multiple Plots
plot_save_loc="X:/SCRIPTS/Python ML/TEST/PDF Output"
plot_name=plot_save_loc+"/Tensorflow_model_outputs_"+str( datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))+".pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(plot_name)

with pdf:
    if train_data_type.upper() == "SIMPLE":

        for i,j,k,z in zip(x_plot_test,y_plot_test,test_y_pred,test_input_ls):


            plt.figure(figsize=(11.69,8.27))
            sp1 = plt.subplot(1, 2, 1)
            # sp1.set_title("Forecast vs Actual", fontsize=14)
            sp1.set_title("Prod Comparison - Noise Pct " + str(round(z[3],2)*100)+"%", fontsize=14)
            sp1.plot(pd.Series(np.ravel(i)), "bd", markersize=10, label="noisy", alpha=0.4)
            sp1.plot(pd.Series(np.ravel(j)), "r*", markersize=10, label="Theoretical Best", alpha=1)
            sp1.plot(pd.Series(np.ravel(k)), "g.", markersize=10, label="Forecast", alpha=0.7)
            sp1.legend(loc="upper right")
            # sp1.legend([str(round(z[0], 0)), str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            sp1.text(20,0.5,"IP " + str(round(z[0], 0)) + ", Secant " + str(round(z[1], 2)*100) + "%, B "+ str(round(z[2], 2)))
            sp1.set_xlabel("Time (mos)")
            sp1.set_ylabel("Production Amount")
            sp2 = plt.subplot(1, 2, 2)
            # sp2.set_title("Forecast vs Actual", fontsize=14)
            sp2.set_title("Prod Comparison - Noise Pct " + str(round(z[3],2)*100)+"%", fontsize=14)
            sp2.plot(pd.Series(np.ravel(i)), "bd", markersize=10, label="noisy", alpha=0.4)
            sp2.plot(pd.Series(np.ravel(j)), "r*", markersize=10, label="Theoretical Best", alpha=1)
            sp2.plot(pd.Series(np.ravel(k)), "g.", markersize=10, label="Forecast", alpha=0.7)
            sp2.legend(loc="upper right")
            # sp2.legend([str(round(z[0], 0)),str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            # sp2.legend([str(round(z[0], 0)), str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            # sp2.text(2,0.5,"IP "+str(round(z[0], 0))+", Secant "+str(round(z[1], 0))+", B "+str(round(z[2], 0)))


            sp2.set_xlabel("Time (mos)")
            sp2.set_ylabel("Production Amount")
            sp2.set_yscale('log')
            plt.tight_layout()
            # plt.show()
            pdf.savefig()
            plt.close()

    else:
        for i, j, k, z in zip(x_plot_test, y_plot_test, test_y_pred, test_input_ls):
            plt.figure(figsize=(11.69, 8.27))
            sp1 = plt.subplot(1, 2, 1)
            # sp1.set_title("Forecast vs Actual", fontsize=14)
            sp1.set_title("Prod Comparison - Noise Pct " + str(round(z[4], 2) * 100) + "%", fontsize=14)
            sp1.plot(pd.Series(np.ravel(i)), "bd", markersize=10, label="noisy", alpha=0.4)
            sp1.plot(pd.Series(np.ravel(j)), "r*", markersize=10, label="Theoretical Best", alpha=1)
            sp1.plot(pd.Series(np.ravel(k)), "g.", markersize=10, label="Forecast", alpha=0.7)
            sp1.legend(loc="upper right")
            # sp1.legend([str(round(z[0], 0)), str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            if z[0]==1:
                dca_type="Hyp 2 Exp"
            elif z[0]==2:
                dca_type="Exp"
            elif z[0] == 3:
                dca_type="Flat 2 Hyp 2 Exp"


            sp1.text(20, 0.5,"DCA - "+dca_type+ ":IP " + str(round(z[1], 0)) + ", Secant " + str(round(z[2], 2) * 100) + "%, B " + str(
                round(z[3], 2)))
            sp1.set_xlabel("Time (mos)")
            sp1.set_ylabel("Production Amount")
            sp2 = plt.subplot(1, 2, 2)
            # sp2.set_title("Forecast vs Actual", fontsize=14)
            sp2.set_title("Prod Comparison - Noise Pct " + str(round(z[4], 2) * 100) + "%", fontsize=14)
            sp2.plot(pd.Series(np.ravel(i)), "bd", markersize=10, label="noisy", alpha=0.4)
            sp2.plot(pd.Series(np.ravel(j)), "r*", markersize=10, label="Theoretical Best", alpha=1)
            sp2.plot(pd.Series(np.ravel(k)), "g.", markersize=10, label="Forecast", alpha=0.7)
            sp2.legend(loc="upper right")
            # sp2.legend([str(round(z[0], 0)),str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            # sp2.legend([str(round(z[0], 0)), str(round(z[1], 2)), str(round(z[2], 2)), str(round(z[4], 2))],["IP", "Secant", "B", "Terminal De"], loc="lower right")
            # sp2.text(2,0.5,"IP "+str(round(z[0], 0))+", Secant "+str(round(z[1], 0))+", B "+str(round(z[2], 0)))


            sp2.set_xlabel("Time (mos)")
            sp2.set_ylabel("Production Amount")
            sp2.set_yscale('log')
            plt.tight_layout()
            # plt.show()
            pdf.savefig()
            plt.close()

# pdf.close()



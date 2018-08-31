import sys
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

from sklearn.model_selection import train_test_split


# >>> Read global data
strokes = np.load('../data/strokes.npy', encoding='bytes')
with open('../data/sentences.txt') as f:
    texts = f.readlines()
# <<< Read global data


def load_data(timesteps=700, validation_split=0.3):
    # Input:
    #   timesteps - int
    #   validation_split - float

    # TODO: iterate through timesteps, too

    X_data = np.zeros(shape=(len(strokes), timesteps, 3), dtype=np.float32)
    y_data = np.zeros(shape=(len(strokes), timesteps, 3), dtype=np.float32)

    for i, stroke in enumerate(strokes):
        X_data[i] = stroke[:timesteps]
        y_data[i] = stroke[1 : timesteps + 1]

    return train_test_split(X_data, y_data, test_size=validation_split)


class LstmModel:

    def __init__(self, session, checkpoint, rnn, timesteps, n_input, n_hidden, n_output):
        # Input:
        #   session - TensorFlow Session
        #   checkpoint - str
        #   rnn - TensorFlow RNN Layer loader
        #   timesteps  - int
        #   n_input    - int
        #   n_hidden   - int
        #   n_output   - int

        self.sess = session
        self.checkpoint = checkpoint

        # The amount of sequences fed to the network while training
        self.training_steps = 1000
        self.batch_size = 2
        self.learning_rate = 1e-3

        self.xtr, self.xval, self.ytr, self.yval = load_data(timesteps=timesteps,
            validation_split=0.1)

        with tf.variable_scope("some_scope_name"):
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate)

            # Input placeholder
            self.x = tf.placeholder(np.float32,
                shape=(None, None, n_input), name="input")

            self.y_pred = tf.placeholder(tf.float32, (None, None, n_output))
            self.y_pred_label = tf.reshape(self.y_pred, [-1, n_output])

            # RNN output node weights and biases
            self.weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
            self.biases = tf.Variable(tf.random_normal([n_output]))

            self.rnn_layer = rnn(self.x, self.weights, self.biases, timesteps,
                n_input, n_hidden, n_output)


    def __does_checkpoint_exist(self, checkpoint):
        # Input:
        #   checkpoint - str

        # Output:
        #   exist - bool
        return False


    def __generate_next_batch(self):

        # TODO: redesign for more randomness (obviously)
        random_idx = np.random.randint(0, len(self.xtr) - self.batch_size)
        return self.xtr[random_idx : random_idx + self.batch_size], \
            self.ytr[random_idx : random_idx + self.batch_size]


    def train(self):
        current_step = 1

        while True:
            x_batch, y_batch = self.__generate_next_batch()

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.rnn_layer,
                labels=self.y_pred_label))

            train_op = self.optimizer.minimize(loss_op)

            sys.stdout.write('\repoch %d starting... ' % current_step)

            _, loss = self.sess.run([train_op, loss_op],
                feed_dict={self.x: x_batch, self.y_pred: y_batch})

            sys.stdout.write('done!')
            sys.stdout.flush()

            if current_step == self.training_steps:
                break
            else:
                current_step += 1

            # prediction = tf.nn.softmax(logits)


    def sample(self):
        if not __does_checkpoint_exist(checkpoint):
            # Create the checkpoint first of all
            train()


def get_unconditional_rnn(x, weights, biases, timesteps=700, n_input=3,
                          n_hidden=400, n_output=3):

    x = tf.reshape(x, [-1, n_input])

    # # Generate a n_input-element sequence of inputs
    x = tf.split(x, n_input, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases



def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    checkpoint = '../data/checkpoints/model-prediction.ckpt'

    # We use a single LSTM layer with 900 hidden cells 
    n_hidden = 900

    # Input shape: (stop_sign, x, y)
    n_input = 3
    n_output = n_input

    # We want to output only one (next) point per time
    timesteps_output = 1

    # Take the minimum length over all sequences, in this case
    # we will not get rid of any stroke
    timesteps = np.min([len(x) for x in strokes]) - timesteps_output

    with tf.device('/gpu:0'):
            # To avoid using the totality of the GPU memory
            session_config = tf.ConfigProto(
                allow_soft_placement = True,
                log_device_placement = False,
                gpu_options = tf.GPUOptions(
                    allow_growth = True,
                    per_process_gpu_memory_fraction = 0.7
                )
            )

            with tf.Session(config=session_config) as sess:
                sess.run(tf.global_variables_initializer())

                model = LstmModel(sess, checkpoint, get_unconditional_rnn, timesteps,
                    n_input, n_hidden, n_output)

                model.train()

    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    checkpoint_file = '../data/checkpoints/model-synthesis.ckpt'

    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str

    checkpoint_file = '../data/checkpoints/model-recognition.ckpt'

    return 'welcome to lyrebird'

import sys
import numpy as np
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import rnn

from sklearn.model_selection import train_test_split


# >>> Read global data
strokes = np.load('../data/strokes.npy', encoding='bytes')
with open('../data/sentences.txt') as f:
    texts = f.readlines()
# <<< Read global data


def load_data_predict(timesteps=700, max_samples_per_stroke=200, validation_size=0.3):
    # Input:
    #   timesteps - int
    #   validation_size - float or int

    max_generated_strokes = np.sum(
        [min(len(stroke) - timesteps, max_samples_per_stroke) for stroke in strokes])

    X_data = np.zeros(shape=(max_generated_strokes, timesteps, 3), dtype=np.float32)
    y_data = np.zeros(shape=(max_generated_strokes, timesteps, 3), dtype=np.float32)

    current_timestep = 0

    for stroke in tqdm(strokes):
        if timesteps > len(stroke):
            continue

        possible_ids_num = len(stroke) - timesteps
        selected_ids_num = min(possible_ids_num, max_samples_per_stroke)
        selected_ids = np.random.permutation(possible_ids_num)[:selected_ids_num]

        for stroke_position in selected_ids:
            X_data[current_timestep] = \
                stroke[stroke_position : timesteps + stroke_position]
            # shift to 1 timestep forward
            y_data[current_timestep] = \
                stroke[stroke_position + 1 : timesteps + stroke_position + 1]
            current_timestep += 1

    # TODO: normalize [x, y] to mean=0, stddev=1

    return train_test_split(X_data, y_data, test_size=validation_size)


def load_data_recognize(timesteps=700, validation_size=0.3):
    # Input:
    #   timesteps - int
    #   validation_split - float

    # The function to read strokes and predtct text

    return None


class LstmModel:

    def __init__(self, checkpoint, timesteps, n_input,
        n_hidden, n_output, scope_name="scope"):
        # Input:
        #   session - TensorFlow Session
        #   checkpoint - str
        #   rnn - TensorFlow RNN Layer loader
        #   timesteps  - int
        #   n_input    - int
        #   n_hidden   - int
        #   n_output   - int

        # The amount of sequences fed to the network while training
        self.training_steps = 1000
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.weigths_stddev = 0.075
        self.mixtures = 20

        self.sess = None # Init an empty TF session
        self.checkpoint = checkpoint
        self.scope_name = scope_name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.timesteps = timesteps
        self.n_output = n_output

        self.__initialize_variables()

        session_config = tf.ConfigProto(
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions(
                allow_growth = True,
                # To avoid using the totality of the GPU memory
                per_process_gpu_memory_fraction = 0.7
            )
        )

        self.sess = tf.Session(config=session_config)

        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())


    def __del__(self):
        self.sess.close()


    def __initialize_variables(self):
        with tf.variable_scope(self.scope_name):
            # Input placeholder
            self.x = tf.placeholder(np.float32, shape=(None, None, self.n_input))
            self.y_pred = tf.placeholder(tf.float32, (None, None, self.n_output))
            self.y_pred_label = tf.reshape(self.y_pred, [-1, self.n_output])

            # RNN output node weights and biases
            self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_output], 
                stddev=self.weigths_stddev))
            self.biases = tf.Variable(tf.random_normal([self.n_output]))

            self.rnn_layer = self.__get_rnn_layer(self.x, self.weights, self.biases)
            self.mdn_layer = self.__get_mixture_density_outputs(self.rnn_layer)
            self.loss_op = self.__get_loss(self.mdn_layer)

            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate)

            self.train_op = self.optimizer.minimize(self.loss_op)


    def __get_rnn_layer(self, inputs, weights, biases):

        with tf.variable_scope(self.scope_name):
            rnn_x = tf.reshape(inputs, [-1, self.n_input])

            # # Generate a n_input-element sequence of inputs
            rnn_x = tf.split(rnn_x, self.n_input, 1)

            # 1-layer LSTM with n_hidden units.
            rnn_cell = rnn.BasicLSTMCell(self.n_hidden)

            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, rnn_x, dtype=tf.float32)

            return tf.matmul(outputs[-1], weights) + biases


    def __get_mixture_density_outputs(self, inputs):

        with tf.variable_scope(self.scope_name):
            e = tf.layers.dense(inputs, 1,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))
            pi = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))

            mean1 = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))
            mean2 = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))

            std1 = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))
            std2 = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))

            rho = tf.layers.dense(inputs, self.mixtures,
                kernel_initializer=tf.random_normal_initializer(stddev=0.075))

            e_out = tf.sigmoid(-e)
            pi_out = tf.nn.softmax(pi)
            std1_out = tf.exp(std1)
            std2_out = tf.exp(std2)
            rho_out = tf.tanh(rho)

            return e_out, pi_out, mean1, mean2, std1_out, std2_out, rho_out


    def __get_loss(self, mdn_layer):

        eps = 1e-7

        e, pi, mean1, mean2, std1, std2, rho = mdn_layer

        with tf.variable_scope(self.scope_name):
            x_coords, y_coords, stop_flags = \
                tf.unstack(tf.expand_dims(self.y_pred_label, axis=2), axis=1)

            x_standardized = (x_coords - mean1) / std1
            y_standardized = (y_coords - mean2) / std2

            z = tf.square(x_standardized) + tf.square(x_standardized) - \
                2.0 * rho * x_standardized * y_standardized

            rho_processed = 1 - tf.square(rho)

            n = 1.0 / (2 * np.pi * std1 * std2 * tf.sqrt(rho_processed)) * \
                tf.exp(-z / (2 * rho_processed))

            reduct_sum_pi_n = tf.reduce_sum(pi * n, axis=1)

            e_conditional = tf.multiply(stop_flags, e) + \
                tf.multiply(1. - stop_flags, 1. - e)

            loss = tf.reduce_mean(-tf.log(tf.maximum(reduct_sum_pi_n, eps)) - \
                tf.log(tf.maximum(e_conditional, eps)))

            return loss


    def __does_checkpoint_exist(self, checkpoint):
        # Input:
        #   checkpoint - str

        # Output:
        #   exist - bool
        return False


    def __generate_next_batch(self, X_data, y_data):

        # TODO: redesign for more randomness (obviously)
        random_idx = np.random.randint(0, len(X_data) - self.batch_size)
        return X_data[random_idx : random_idx + self.batch_size], \
            y_data[random_idx : random_idx + self.batch_size]


    def train(self, data_loader):
        current_step = 1
        loss_decreases_num = 0
        validation_step = 10
        validation_best_steps = 2
        best_validation_loss = np.inf

        print("Loading training and validation data...")
        xtr, xval, ytr, yval = data_loader(timesteps=self.timesteps,
            max_samples_per_stroke=25, validation_size=200)

        print("Starting model training with batch size of {}...".format(
                self.batch_size))

        while True:
            start_time = time.time()
            x_batch, y_batch = self.__generate_next_batch(xtr, ytr)

            _, loss = self.sess.run([self.train_op, self.loss_op],
                feed_dict={self.x: x_batch, self.y_pred: y_batch})

            iteration_time = int(time.time() - start_time)

            print("iteration {:>4}: loss {:0.10f} time {:3d}s".format(
                current_step, loss, iteration_time))

            if current_step % validation_step == 0:
                val_loss = self.sess.run(self.loss_op,
                    feed_dict={self.x: xval, self.y_pred: yval})

                print("val_loss {:0.10f}".format(val_loss))

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    loss_decreases_num = 0
                    self.saver.save(self.sess, self.checkpoint)
                else:
                    loss_decreases_num += 1

            if current_step == self.training_steps or 
                    loss_decreases_num == validation_best_steps:
                break
            else:
                current_step += 1


    def sample(self):
        # if not __does_checkpoint_exist(self.checkpoint):
        #     # Create the checkpoint first of all or raise an exception
        #     pass
        # else:

        self.saver.restore(self.sess, self.checkpoint)

        starting_step = np.array([0., 0., 1.])


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    checkpoint = '../data/checkpoints/model-prediction.ckpt'

    # We use a single LSTM layer with 900 hidden cells 
    n_hidden = 900

    # Input shape: (x, y, stop_sign)
    n_input = 3
    n_output = n_input

    # We want to output only one (next) point per time
    timesteps_output = 1

    # Take the minimum length over all sequences, in this case
    # we will not get rid of any stroke
    timesteps = np.min([len(x) for x in strokes]) - timesteps_output

    model = LstmModel(checkpoint, timesteps,
        n_input, n_hidden, n_output, scope_name="lstm_unconditional")

    model.sample()

    # model.train(load_data_predict)

    return None


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

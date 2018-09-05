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
        self.training_steps = 10000
        self.batch_size = 64
        self.weigths_stddev = 0.075
        self.weights_mean = 0
        self.mixtures = 20

        self.learning_rate = 1e-4
        self.optimizer_decay = 0.95
        self.optimizer_momentum = 0.9
        self.gradients_clip = 10

        self.sess = None # Init an empty TF session
        self.checkpoint = checkpoint
        self.scope_name = scope_name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.timesteps = timesteps
        # pi, mean1, mean2, std1, std2, rho + "end of stroke"
        self.n_output = self.mixtures * 6 + 1

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
            self.y_pred = tf.placeholder(tf.float32, (None, None, self.n_input))
            self.y_pred_label = tf.reshape(self.y_pred, [-1, self.n_input])

            # RNN output node weights and biases
            self.weights = tf.Variable(tf.random_normal(
                [self.n_hidden, self.n_output],
                mean=self.weights_mean,
                stddev=self.weigths_stddev))

            self.biases = tf.Variable(tf.random_normal([self.n_output],
                mean=self.weights_mean, stddev=self.weigths_stddev))

            self.rnn_layer = self.__get_rnn_layer(self.x, self.weights, self.biases)
            self.mdn_layer = self.__get_mixture_density_outputs(self.rnn_layer)
            self.loss_op = self.__get_loss(self.mdn_layer)

            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
                decay=self.optimizer_decay,
                momentum=self.optimizer_momentum)

            training_vars = tf.trainable_variables()
            gradients, _ = \
                tf.clip_by_global_norm(tf.gradients(self.loss_op, training_vars),
                    self.gradients_clip)

            self.train_op = \
                self.optimizer.apply_gradients(zip(gradients, training_vars))


    def __get_rnn_layer(self, inputs, weights, biases):

        with tf.variable_scope(self.scope_name):
            rnn_x = [
                tf.squeeze(input_, [1]) \
                    for input_ in tf.split(inputs, self.timesteps, 1)
            ]

            # 1-layer LSTM with n_hidden units.
            rnn_cell = rnn.BasicLSTMCell(self.n_hidden)

            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, rnn_x, dtype=tf.float32)

            rnn_out = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])

            return tf.matmul(rnn_out, weights) + biases


    def __get_mixture_density_outputs(self, inputs):

        with tf.variable_scope(self.scope_name):
            pi, mean1, mean2, std1, std2, rho = tf.split(inputs[:, 1:], 6, 1)
            e = inputs[:, :-1]

            e_out = tf.sigmoid(-e)
            pi_out = tf.nn.softmax(pi)
            std1_out = tf.exp(std1)
            std2_out = tf.exp(std2)
            rho_out = tf.tanh(rho)

            return e_out, pi_out, mean1, mean2, std1_out, std2_out, rho_out


    def __get_loss(self, mdn_layer):

        eps = 1e-10

        self.e, self.pi, self.mean1, self.mean2, self.std1, self.std2, self.rho = \
            mdn_layer

        with tf.variable_scope(self.scope_name):
            stop_flags, x_coords, y_coords = tf.split(self.y_pred_label, 3, 1)

            x_no_mean = tf.subtract(x_coords, self.mean1)
            y_no_mean = tf.subtract(y_coords, self.mean2)
            x_standardized = tf.div(x_no_mean, self.std1)
            y_standardized = tf.div(y_no_mean, self.std2)

            z = tf.square(x_standardized) + tf.square(y_standardized) - \
                tf.div(2.0 * tf.multiply(self.rho, tf.multiply(x_no_mean, y_no_mean)), \
                tf.multiply(self.std1, self.std2))

            rho_processed = 1.0 - tf.square(self.rho)

            n = tf.div(tf.exp(tf.div(-z, 2.0 * rho_processed)),
                2.0 * np.pi * tf.multiply(tf.multiply(self.std1, self.std2),
                    tf.sqrt(rho_processed)))

            reduct_sum_pi_n = tf.reduce_sum(tf.multiply(self.pi, n), axis=1,
                keep_dims=True)

            e_conditional = tf.multiply(stop_flags, self.e) + \
                tf.multiply(1.0 - stop_flags, 1.0 - self.e)

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
        validation_step = 100
        validation_best_steps = 10
        best_validation_loss = np.inf

        print("Loading training and validation data...")
        xtr, xval, ytr, yval = data_loader(timesteps=self.timesteps,
            max_samples_per_stroke=25, validation_size=self.batch_size)

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

            if current_step == self.training_steps or \
                    loss_decreases_num == validation_best_steps:
                break
            else:
                current_step += 1


    def __create_point(self, e, mean1, mean2, std1_, std2_, rho):
        max_val = np.float32(1e+5)

        # There are huge numbers sometimes and numpy returns inf when casting
        std1 = np.minimum(max_val, std1_)
        std2 = np.minimum(max_val, std2_)

        covariance_matrix = np.array([[std1 * std1, std1 * std2 * rho],
            [std1 * std2 * rho, std2 * std2]])

        mean = np.array([mean1, mean2])

        x, y = np.random.multivariate_normal(mean, covariance_matrix)
        return np.array([x, y, np.float32(e > 0.5)])


    def sample(self, timesteps=700, from_text="", random_seed=1):
        # if not __does_checkpoint_exist(self.checkpoint):
        #     # Create the checkpoint first of all or raise an exception
        #     pass
        # else:

        np.random.seed(random_seed)

        self.saver.restore(self.sess, self.checkpoint)
        output_sequence = np.zeros((timesteps, 3))

        # initialize with the point at (0, 0)
        current_point = np.array([1.0, 0., 0.0], dtype=np.float32)
        output_sequence[0] = current_point

        for sequence_step in tqdm(range(1, timesteps)):
            e, pi, mean1, mean2, std1, std2, rho = self.sess.run(
                [
                    self.e, self.pi, self.mean1, self.mean2,
                    self.std1, self.std2, self.rho
                ],
                feed_dict={self.x: current_point[None, None, ...]})

            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])

            new_point = self.__create_point(e[0, 0], mean1[0, g],
                mean2[0, g], std1[0, g], std2[0, g], rho[0, g])

            current_point = new_point
            output_sequence[sequence_step] = current_point

        # end of stroke
        output_sequence[-1, 0] = 1.0

        return output_sequence


def generate_unconditionally(random_seed=1, mode='sample'):
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
    timesteps = np.min([len(x) for x in strokes]) - timesteps_output \
        if mode == 'train' else timesteps_output

    model = LstmModel(checkpoint, timesteps,
        n_input, n_hidden, n_output, scope_name="lstm_unconditional")

    if mode == 'train':
        model.train(load_data_predict)
    else:
        return model.sample(timesteps=700, random_seed=random_seed)

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

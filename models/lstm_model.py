from __future__ import print_function

import os
import sys
import errno
import numpy as np
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import rnn

from sklearn.model_selection import train_test_split

# >>> Read global data
strokes = np.load("../data/strokes.npy", encoding="bytes")
with open("../data/sentences.txt") as f:
    texts = f.readlines()
# <<< Read global data


def load_data_predict(timesteps=700,
                      max_samples_per_stroke=200,
                      validation_size=0.3):
    """Generates training and validation datasets using the global strokes data

    Args:
        timesteps (int, optional): Length of the data sequence (stroke)
        max_samples_per_stroke (int, optional)
        validation_size (float or int, optional): Train-test split ratio (float) or a test set size (int)

    Returns:
        4 arrays: X_train, X_test, y_train, y_test
    """

    max_generated_strokes = np.sum([
        min(len(stroke) - timesteps, max_samples_per_stroke)
        for stroke in strokes
    ])

    X_data = np.zeros(
        shape=(max_generated_strokes, timesteps, 3), dtype=np.float32)
    y_data = np.zeros(
        shape=(max_generated_strokes, timesteps, 3), dtype=np.float32)

    current_timestep = 0

    for stroke in tqdm(strokes):
        if timesteps > len(stroke):
            continue

        possible_ids_num = len(stroke) - timesteps
        selected_ids_num = min(possible_ids_num, max_samples_per_stroke)
        selected_ids = np.random.permutation(
            possible_ids_num)[:selected_ids_num]

        for stroke_position in selected_ids:
            X_data[current_timestep] = \
                stroke[stroke_position : timesteps + stroke_position]
            # shift to 1 timestep forward
            y_data[current_timestep] = \
                stroke[stroke_position + 1 : timesteps + stroke_position + 1]
            current_timestep += 1

    return train_test_split(X_data, y_data, test_size=validation_size)


def load_data_recognize(timesteps=700, validation_size=0.3):
    """Generates training and validation data for handwriting prediction purposes

    Args:
        timesteps (int, optional): Length of the data sequence
        validation_size (float or int, optional): Train-test split ratio (float) or a test set size (int)

    Returns:
        4 arrays: X_train, X_test, y_train, y_test
    """
    pass


class LstmModel:
    """LSTM-based RNN model with a Mixture Density Outputs layer
    """

    def __init__(self,
                 checkpoint,
                 timesteps,
                 n_input,
                 n_hidden,
                 scope_name="scope"):
        """Summary

        Args:
            checkpoint (str): Model checkpoint filepath
            timesteps (int): Length of the data sequence (stroke)
            n_input (int): Size of the input stroke point, [stop_sign, x, y]
            n_hidden (int): Number of cells in the RNN layer
            n_output (int): Size of the last output layer
            scope_name (str, optional): Name of the TensorFlow variables scope
        """

        self.epochs = 1000
        self.steps_per_epoch = 100
        self.batch_size = 64
        self.weigths_stddev = 0.075
        self.weights_mean = 0
        self.mixtures = 20

        self.learning_rate = 1e-4
        self.optimizer_decay = 0.95
        self.optimizer_momentum = 0.9
        self.gradients_clip = 10

        self.checkpoint = checkpoint
        self.scope_name = scope_name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.timesteps = timesteps
        self.n_output = self.mixtures * 6 + 1  # pi, mean1, mean2, std1, std2, rho + "end of stroke"

        self._initialize_variables()

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                allow_growth=True,
                # To avoid using the totality of the GPU memory
                per_process_gpu_memory_fraction=0.7))

        self.sess = tf.Session(config=session_config)

        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def __del__(self):
        """TensorFlow Session destructor
        """
        self.sess.close()

    def _initialize_variables(self):
        """Initialize tf Variables and operations
        """
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            # Input placeholder
            self.x = tf.placeholder(
                np.float32, shape=(None, None, self.n_input))
            self.y_pred = tf.placeholder(
                tf.float32, shape=(None, None, self.n_input))
            self.y_pred_label = tf.reshape(self.y_pred, [-1, self.n_input])

            self.weights = tf.Variable(
                tf.random_normal(
                    [self.n_hidden, self.n_output],
                    mean=self.weights_mean,
                    stddev=self.weigths_stddev))

            self.biases = tf.Variable(
                tf.random_normal(
                    [self.n_output],
                    mean=self.weights_mean,
                    stddev=self.weigths_stddev))

            self.rnn_layer = self._get_rnn_layer(self.x, self.weights,
                                                 self.biases)
            self.mdn_layer = self._get_mixture_density_outputs(self.rnn_layer)
            self.loss_op = self._get_loss(self.mdn_layer)

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

    def _get_rnn_layer(self, inputs, weights, biases):
        """Get the RNN layer with the structure of LSTM cells

        Args:
            inputs (tf.Tensor)
            weights (tf.Variable)
            biases (tv.Variable)

        Returns:
            tf.Tensor: RNN layer
        """
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
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

    def _get_mixture_density_outputs(self, inputs):
        """Use the outputs of a neural network to parameterise a mixture distribution

        e: end of stroke probability
        pi: mixture weights
        rho: correlations of the mixture components

        Args:
            inputs (tf.Tensor): RNN layer

        Returns:
            6 x tf.Tensor: 
        """
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            pi, mean1, mean2, std1, std2, rho = tf.split(inputs[:, 1:], 6, 1)
            e = inputs[:, :1]

            self.pi = pi
            self.std1 = std1
            self.std2 = std2

            e_out = tf.sigmoid(-e)
            pi_out = tf.nn.softmax(pi)
            mean1_out = mean1
            mean2_out = mean2
            std1_out = tf.exp(std1)
            std2_out = tf.exp(std2)
            rho_out = tf.tanh(rho)

            return [e_out, pi_out, mean1_out, mean2_out, std1_out, std2_out, rho_out]

    def _get_loss(self, mdn_layer):
        """Calculate the network loss according to RNN and MDN layer outputs

        Args:
            mdn_layer (6 x tf.Tensor): result of the Mixture Density Outputs layer

        Returns:
            tf.Tensor: Reduced tf Tensor
        """
        eps = 1e-10

        [e, pi, mean1, mean2, std1, std2, rho] = mdn_layer

        self.e = e
        self.mean1 = mean1
        self.mean2 = mean2
        self.rho = rho

        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            stop_flags, x_coords, y_coords = tf.split(self.y_pred_label, 3, 1)

            x_no_mean = tf.subtract(x_coords, mean1)
            y_no_mean = tf.subtract(y_coords, mean2)
            x_standardized = tf.div(x_no_mean, std1)
            y_standardized = tf.div(y_no_mean, std2)

            z = tf.square(x_standardized) + tf.square(y_standardized) - \
                tf.div(2.0 * tf.multiply(rho, tf.multiply(x_no_mean, y_no_mean)), \
                tf.multiply(std1, std2))

            rho_processed = 1.0 - tf.square(rho)

            n = tf.div(
                tf.exp(tf.div(-z, 2.0 * rho_processed)),
                2.0 * np.pi * tf.multiply(
                    tf.multiply(std1, std2), tf.sqrt(rho_processed)))

            reduct_sum_pi_n = tf.reduce_sum(
                tf.multiply(pi, n), axis=1, keep_dims=True)

            e_conditional = tf.multiply(stop_flags, e) + \
                tf.multiply(1.0 - stop_flags, 1.0 - e)

            loss = tf.reduce_mean(-tf.log(tf.maximum(reduct_sum_pi_n, eps)) - \
                tf.log(tf.maximum(e_conditional, eps)))

            return loss

    def _generate_next_batch(self, X_data, y_data):
        """Generate next batch of training data

        Args:
            X_data (np.array)
            y_data (np.array)

        Returns:
            np.array: A batch of size `self.batch_size`
        """
        # TODO: redesign for more randomness (obviously)
        random_idx = np.random.randint(0, len(X_data) - self.batch_size)
        return X_data[random_idx : random_idx + self.batch_size], \
            y_data[random_idx : random_idx + self.batch_size]

    def _validate_batch(self, X_data, y_data):
        """Calculate the loss for the validation set
        
        Args:
            X_data (np.array)
            y_data (np.array)
        
        Returns:
            float: validation loss
        """
        num_iterations = len(X_data) // self.batch_size
        losses = np.zeros((num_iterations), dtype=np.float32)

        for batch_id in range(num_iterations):
            id_start = batch_id * self.batch_size
            id_end = id_start + self.batch_size

            x_batch = X_data[id_start:id_end]
            y_batch = y_data[id_start:id_end]

            loss = self.sess.run(
                self.loss_op,
                feed_dict={
                    self.x: x_batch,
                    self.y_pred: y_batch
                })

            losses[batch_id] = loss

        return np.mean(losses)

    def _create_point(self, e, mean1, mean2, std1, std2, rho):
        """Sample a stroke point from the network outputs

        Args:
            e (int): End of stroke probability
            mean1 (int)
            mean2 (int)
            std1_ (int)
            std2_ (int)
            rho (int): Correlations of the mixture components 

        Returns:
            np.array: [self.n_input] shape single stroke point
        """
        covariance_matrix = np.array([[std1 * std1, std1 * std2 * rho],
                                      [std1 * std2 * rho, std2 * std2]])

        mean = np.array([mean1, mean2])

        x = np.random.multivariate_normal(mean, covariance_matrix, 1)
        return np.array([np.float32(e > 0.07), x[0][0], x[0][1]])

    def train(self, data_loader):
        """Train the network on the handwriting data

        Args:
            data_loader (void): a data loading function callback
        """
        loss_degradation_num = 0
        validation_best_steps = 10
        best_validation_loss = np.inf

        print("Generating training and validation data...")
        xtr, xval, ytr, yval = data_loader(
            timesteps=self.timesteps,
            max_samples_per_stroke=50,
            validation_size=0.05)

        print("Starting model training with batch size of {}...".format(
            self.batch_size))

        if tf.train.checkpoint_exists(self.checkpoint):
            self.saver.restore(self.sess, self.checkpoint)

        for current_epoch in range(self.epochs):
            start_time = time.time()

            for current_iteration in range(self.steps_per_epoch):

                x_batch, y_batch = self._generate_next_batch(xtr, ytr)

                _, loss = self.sess.run(
                    [self.train_op, self.loss_op],
                    feed_dict={
                        self.x: x_batch,
                        self.y_pred: y_batch
                    })

                iteration_time = int(time.time() - start_time)
                current_progress = current_iteration % self.steps_per_epoch + 1

                sys.stdout.write(
                    "\r epoch: {:>4d}, progress: {:>3d}%, loss: {:0.10f}, time {:3d}s".
                    format(current_epoch + 1, current_progress, loss,
                           iteration_time))
                sys.stdout.flush()

            val_loss = self._validate_batch(xval, yval)
            sys.stdout.write(" | val_loss: {:0.10f}   \n".format(val_loss))

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                loss_degradation_num = 0
                self.saver.save(self.sess, self.checkpoint)
            else:
                loss_degradation_num += 1

            if loss_degradation_num == validation_best_steps:
                break

    def sample(self, timesteps=700, random_seed=1):
        """Generate a random stroke from a (0, 0) starting point

        Args:
            timesteps (int, optional): Length of the data sequence (stroke)
            random_seed (int, optional)

        Returns:
            np.array: sampled stroke of a shape (timesteps, n_input) 

        Raises:
            FileNotFoundError: TensorFlow model checkpoint does not exist
        """
        sample_bias = 0.15

        if not tf.train.checkpoint_exists(self.checkpoint):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.checkpoint)

        np.random.seed(random_seed)

        self.saver.restore(self.sess, self.checkpoint)
        output_sequence = np.zeros((timesteps, 3), dtype=np.float32)

        # initialize with the point at (0, 0)
        current_point = np.array([1.0, 0., 0.0], dtype=np.float32)
        output_sequence[0] = current_point

        for sequence_step in tqdm(range(1, timesteps)):
            e, pi, mean1, mean2, std1, std2, rho = self.sess.run(
                [
                    self.e, self.pi, self.mean1, self.mean2, self.std1,
                    self.std2, self.rho
                ],
                feed_dict={self.x: current_point[None, None, ...]})

            # Biased sampling
            std1 = np.exp(std1 - sample_bias)
            std2 = np.exp(std2 - sample_bias)
            pi_ = pi * (1.0 + sample_bias)
            pi = np.zeros_like(pi_)
            pi[0] = np.exp(pi_[0]) / np.sum(np.exp(pi_[0]), axis=0)

            pi_idx = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            new_point = self._create_point(e[0, 0], mean1[0, pi_idx], 
                                           mean2[0, pi_idx], std1[0, pi_idx],
                                           std2[0, pi_idx], rho[0, pi_idx])

            current_point = new_point
            output_sequence[sequence_step] = current_point

        # end of stroke
        output_sequence[-1, 0] = 1.0

        return output_sequence


def generate_unconditionally(random_seed=1, mode="sample"):
    """Summary

    Args:
        random_seed (int, optional)
        mode (str, optional): Either `sample` or `train`

    Returns:
        np.array: Generated stroke with the shape (timesteps, n_input)
    """
    checkpoint = "../data/checkpoints/model-prediction.ckpt"

    # We use a single LSTM layer with 900 hidden cells
    n_hidden = 900

    # Input shape: (stop_sign, x, y)
    n_input = 3

    # We want to output only one (next) point per time
    timesteps_output = 1

    # Take the minimum length over all sequences, in this case
    # we will not get rid of any stroke
    timesteps = np.min([len(x) for x in strokes]) - timesteps_output \
        if mode == "train" else timesteps_output

    model = LstmModel(
        checkpoint,
        timesteps,
        n_input,
        n_hidden,
        scope_name="lstm_unconditional")

    if mode == "train":
        model.train(load_data_predict)
        return None
    else:
        return model.sample(timesteps=700, random_seed=random_seed)


def generate_conditionally(text="welcome to lyrebird", random_seed=1):
    """

    Args:
        text (str, optional): Description
        random_seed (int, optional): Description

    Returns:
        np.array: stroke - numpy 2D-array (T x 3)
    """
    checkpoint_file = "../data/checkpoints/model-synthesis.ckpt"

    return np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def recognize_stroke(stroke):
    """

    Args:
        stroke (np.array): numpy 2D-array (T x 3)

    Returns:
        str: Recognized text
    """
    checkpoint_file = "../data/checkpoints/model-recognition.ckpt"

    return "welcome to lyrebird"

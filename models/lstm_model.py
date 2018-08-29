import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# >>> Read global data
strokes = np.load('../data/strokes.npy')
with open('../data/sentences.txt') as f:
    texts = f.readlines()
# <<< Read global data


def load_data(timesteps=700, validation_split=0.3):
    # Input:
    #   timesteps - int
    #   validation_split - float

    return xtr, ytr, xval, yval


def LstmModel():

    def __init__(self, checkpoint, rnn, timesteps, n_input, n_hidden, n_output):
        # Input:
        #   checkpoint - str
        #   rnn - TensorFlow RNN Layer loader
        #   timesteps  - int
        #   n_input    - int
        #   n_hidden   - int
        #   n_output   - int

        self.checkpoint = checkpoint

        # The amount of sequences fed to the network while training
        self.training_steps = 1000
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate)


        # Input placeholder
        self.x = tf.placeholder(np.float32,
            shape=(None, None, timesteps, n_input), name="input")

        self.y_pred = tf.placeholder(tf.float32, (None, None, self.n_output))
        self.y_pred_label = tf.reshape(self.y_batch, [-1, self.n_output])

        # RNN output node weights and biases
        self.weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
        self.biases = tf.Variable(tf.random_normal([n_output]))

        self.rnn_layer = rnn(self.x, self.weights, self.biases, timesteps,
            n_input, n_hidden, n_output)

        self.xtr, self.ytr, self.xval, self.yval = load_data(timesteps=timesteps,
            validation_split=0.1)



    def __does_checkpoint_exist(self, checkpoint):
        # Input:
        #   checkpoint - str

        # Output:
        #   exist - bool
        return False


    def __generate_next_batch(self):



    def train(self):
        current_step = 1

        while True:
            x_batch, y_batch = self.__generate_next_batch()

            loss_op = tf.reduce_mean(f.nn.softmax_cross_entropy_with_logits(
                logits=self.rnn_layer,
                labels=self.y_pred_label))

            _, loss = self.session.run([self.optimizer, loss_op],
                feed_dict={self.x: x_batch, self.y: y_batch})

            if current_step == self.training_steps:
                break
            else:
                current_step += 1



    def sample(self):
        if not __does_checkpoint_exist(checkpoint):
            # Create the checkpoint first of all
            train()


def get_unconditional_rnn(x, weights, bias, timesteps=700, n_input=3,
                          n_hidden=400, n_output=3):

    x = tf.reshape(x, [timesteps, n_input])

    # Generate a n_input-element sequence of inputs
    x = tf.split(x, timesteps, 1)

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

    model = get_model(checkpoint, get_unconditional_rnn, timesteps,
        n_input, n_hidden, n_output)

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

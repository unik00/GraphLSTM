import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from pos_utils import *


class CNNCharEmbedding(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: choose these parameters
        self.VOCAB_SIZE = 100
        self.PADDED_LENGTH = 20
        self.CONV_SIZE = 3
        self.NUM_FILTERS = 20
        self.CHAR_DIM = 4

        self.chars_emb = Embedding(self.VOCAB_SIZE, self.CHAR_DIM, input_length=self.PADDED_LENGTH)
        self.conv = Conv2D(kernel_size=(self.CONV_SIZE, 1), strides=1, filters=self.NUM_FILTERS)
        self.max_pool = MaxPooling2D(pool_size=(self.PADDED_LENGTH - self.CONV_SIZE + 1, 1),
                                     strides=1,
                                     data_format='channels_last')

    def call(self, inputs):
        x = self.chars_emb(inputs)
        print(x.shape)
        x = tf.reshape(x, [1, self.PADDED_LENGTH, self.CHAR_DIM, 1])
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class POSEmbedding(Layer):
    """The POS embedding are initialized randomly with a EMB_DIM dimensional vector.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.POS_MAP = get_universal_POS()
        self.NUM_TYPE = len(self.POS_MAP)

        self.EMB_DIM = 10

        self.POS_emb = Embedding(self.NUM_TYPE, self.EMB_DIM, input_length=1)
        # is input_length always 1?

    def call(self, inputs):
        return self.POS_emb(inputs)


class GraphLSTM(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    char_emb = CNNCharEmbedding()
    x = np.random.randint(100, size=20)
    print(x)
    print(char_emb(x))
    pass

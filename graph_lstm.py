from typing import List

import numpy as np
import tensorflow as tf
from allennlp.modules.elmo import batch_to_ids, Elmo
from tensorflow.keras.layers import *

from emb_utils import *


class CNNCharEmbedding(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: keras Hyperband tuner
        self.VOCAB_SIZE = 100

        self.PADDED_LENGTH = 20
        self.CONV_SIZE = 3
        self.NUM_FILTERS = 20
        self.CHAR_DIM = 4

        self.chars_emb = Embedding(self.VOCAB_SIZE, self.CHAR_DIM, input_length=self.PADDED_LENGTH)
        self.conv = Conv2D(kernel_size=(self.CONV_SIZE, self.CHAR_DIM),
                           strides=(1, self.CHAR_DIM),
                           filters=self.NUM_FILTERS)
        self.max_pool = MaxPooling2D(pool_size=(self.PADDED_LENGTH - self.CONV_SIZE + 1, 1),
                                     strides=1,
                                     data_format='channels_last')

    def call(self, inputs):
        """
        Args:
              inputs: a batch of strings denoting tokens
        """
        print("inputs: ", inputs)
        x = self.chars_emb(inputs)

        x = tf.reshape(x, [-1, self.PADDED_LENGTH, self.CHAR_DIM, 1])
        x = self.conv(x)
        x = self.max_pool(x)
        x = tf.reshape(x, [-1, 1, self.NUM_FILTERS])
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

    def call(self, list_POSs: List[str]):
        mapped_inputs = tf.convert_to_tensor([self.POS_MAP[POS] for POS in list_POSs])
        print("mapped inputs: ", mapped_inputs)
        return self.POS_emb(mapped_inputs)


class ELMoEmbedding(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ELMoEmbedding = Elmo("weights/pubmed/pubmed.options",
                                  "weights/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5",
                                  num_output_representations=1,
                                  dropout=0)

    def call(self, inputs):
        """
        Args:
            inputs: a batch of tokenized sentences (strings)
        """
        input_tensor = batch_to_ids(inputs)
        return self.ELMoEmbedding(input_tensor)


class GraphLSTM(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # embeddings
        self.char_emb = CNNCharEmbedding()
        self.elmo_emb = ELMoEmbedding()
        self.pos_emb = POSEmbedding()

    def emb_single(self, sentence: str):
        """ Returns embedding for one sentence
        Args:
            sentence: one tokenized sentence, containing one sentence only
        """
        # get token id from vocab
        doc = gen_dependency_tree(sentence)
        tokens = [token.text for token in doc]

        # get POS
        list_POSs = [token.pos_ for token in doc]

        # concatenate embeddings
        es = self.char_emb(tokens)
        cs = self.elmo_emb([tokens])
        ps = self.pos_emb(list_POSs)
        print(es.shape)
        print(cs.shape)
        print(ps.shape)

    def call(self, inputs):

        pass


if __name__ == "__main__":
    char_emb = CNNCharEmbedding()
    x = np.random.randint(100, size=(2, 20))
    print(x)
    print("char emb x shape: ", char_emb(x).shape)
    # elmo_emb = ELMoEmbedding()
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"
    # print(elmo_emb([s.split(" ")]))

    # pos_emb = POSEmbedding()
    # print(pos_emb(s).shape)
    # model = GraphLSTM()
    # model.emb_single(s)

    pass

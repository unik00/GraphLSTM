from typing import List

import tensorflow as tf
from allennlp.modules.elmo import batch_to_ids, Elmo
from tensorflow.keras.layers import *

from cdr_data import CDRData, normalize
from emb_utils import *


class CNNCharEmbedding(Layer):
    PADDED_LENGTH = 20
    CONV_SIZE = 3
    NUM_FILTERS = 20
    CHAR_DIM = 4

    def __init__(self, char_dict):
        super().__init__(char_dict)
        # TODO: keras Hyperband tuner

        self.char_dict = char_dict

        self.chars_emb = Embedding(len(self.char_dict), self.CHAR_DIM, input_length=self.PADDED_LENGTH)
        self.conv = Conv2D(kernel_size=(self.CONV_SIZE, self.CHAR_DIM),
                           strides=(1, self.CHAR_DIM),
                           filters=self.NUM_FILTERS)
        self.max_pool = MaxPooling2D(pool_size=(self.PADDED_LENGTH - self.CONV_SIZE + 1, 1),
                                     strides=1,
                                     data_format='channels_last')

    def convert_to_ints(self, word):
        ret = list()
        word = normalize(word)
        for c in word:
            ret.append(self.char_dict[c])
        return ret

    def pad(self, emb_chars: List[int]):
        """Center padding one token with 0s
        Args:
            emb_chars: embedded characters of one token
        """
        front = [0] * ((self.PADDED_LENGTH - len(emb_chars)) // 2)
        end = [0] * ((self.PADDED_LENGTH - len(emb_chars) + 1) // 2)
        ret = front + emb_chars + end
        assert len(ret) == self.PADDED_LENGTH
        return ret

    def call(self, inputs: List[str]):
        """
        Args:
              inputs: a batch of strings denoting tokens
        """
        input_tensor = list()

        for s in inputs:
            emb = self.convert_to_ints(s)
            processed_token = self.pad(emb)
            processed_token = tf.convert_to_tensor(processed_token)
            input_tensor.append(processed_token)

        input_tensor = tf.convert_to_tensor(input_tensor)

        x = self.chars_emb(input_tensor)

        x = tf.reshape(x, [-1, self.PADDED_LENGTH, self.CHAR_DIM, 1])
        x = self.conv(x)
        x = self.max_pool(x)
        x = tf.reshape(x, [-1, self.NUM_FILTERS])
        return x


class POSEmbedding(Layer):
    """The POS embedding are initialized randomly with a EMB_DIM dimensional vector.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_map = get_universal_POS()

        self.EMB_DIM = 10

        self.POS_emb = Embedding(len(self.pos_map), self.EMB_DIM, input_length=1)

    def call(self, list_POSs: List[str]):
        mapped_inputs = tf.convert_to_tensor([self.pos_map[POS] for POS in list_POSs])
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
        # what's the mask is there in the dict for?
        return tf.convert_to_tensor(self.ELMoEmbedding(input_tensor)["elmo_representations"][0][0].tolist())


class GraphLSTM(tf.keras.Model):
    BI_LSTM_PHASE_1_OUTPUT_DIM = 30

    def __init__(self, dataset):
        super().__init__()

        # embeddings
        self.dataset = dataset
        self.char_emb = CNNCharEmbedding(self.dataset.char_dict)
        self.elmo_emb = ELMoEmbedding()
        self.pos_emb = POSEmbedding()

        # first BiLSTM phase
        self.biLSTM_1 = Bidirectional(LSTM(self.BI_LSTM_PHASE_1_OUTPUT_DIM,
                                           return_sequences=True,
                                           return_state=False))

    def emb_single(self, sentence: str):
        """ Returns embedding for one sentence
        Args:
            sentence: one tokenized sentence, containing one sentence only
        Returns:
            one tensor of shape [batch_size, d1 + d2 + d3]
            d1, d2, d3 are shape of 3 types of embeddings
        """
        # get token id from vocab
        doc = gen_dependency_tree(sentence)
        tokens = [token.text for token in doc]

        # get POS
        list_POSs = [token.pos_ for token in doc]

        # concatenate embeddings
        ps = self.pos_emb(list_POSs)
        es = self.char_emb(tokens)
        print(tokens)
        cs = self.elmo_emb(list([tokens]))
        print(es.shape)
        print(cs.shape)
        print(ps.shape)
        concatenated = tf.concat([es, cs, ps], 1)
        print(concatenated.shape)

        return concatenated

    def call(self, input):
        """
        Args:
            input: a single string. TODO (after finish): infer by batch
        """
        # TODO: padding before inference?
        output = self.emb_single(input)  # embedding layer

        output = tf.expand_dims(output, axis=0)

        output = self.biLSTM_1(output)
        print("output: ", output)
        print("output.shape: ", output.shape)

        # TODO: graph LSTM

        return output


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"

    data = CDRData()
    char_emb = CNNCharEmbedding(data.char_dict)
    # x = np.random.randint(100, size=(2, 20))
    # print(x)
    # print("char emb x shape: ", char_emb(x).shape)
    # elmo_emb = ELMoEmbedding()
    # print(elmo_emb([s.split(" ")]))
    # pos_emb = POSEmbedding()
    # print(pos_emb(s).shape)
    model = GraphLSTM(data)
    # model.emb_single(s)
    model(s)
    pass

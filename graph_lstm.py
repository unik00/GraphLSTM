from typing import List

import tensorflow as tf
import numpy as np
from allennlp.modules.elmo import batch_to_ids, Elmo
from tensorflow.keras.layers import *

from cdr_data import CDRData, normalize
from emb_utils import *
from graph_lstm_utils import AdjMatrixBuilder


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
    EMB_DIM = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_map = get_universal_POS()

        self.POS_emb = Embedding(len(self.pos_map), self.EMB_DIM, input_length=1)

    def call(self, list_POSs: List[str]):
        mapped_inputs = tf.convert_to_tensor([self.pos_map[POS] for POS in list_POSs])
        # print("POS mapped inputs: ", mapped_inputs)
        return self.POS_emb(mapped_inputs)


class ELMoEmbedding(Layer):
    OUTPUT_DIM = 1024

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


class CalculateHLayer(Layer):
    TRANSITION_STATE_OUTPUTS_DIM = 150

    def __init__(self):
        super(CalculateHLayer, self).__init__()

    def call(self, unpreprocessed_unweight_adj_matrix, h):
        """
        Args:
            input: a string denoting one single sentence
            h: previous hidden state
        """
        leng_doc = h.shape[0]
        # h: 13*150
        unweight_adj_matrix = tf.reshape(unpreprocessed_unweight_adj_matrix, (leng_doc, leng_doc, 2, 1))
        unweight_adj_matrix = tf.broadcast_to(unweight_adj_matrix, (leng_doc,
                                                                    leng_doc,
                                                                    2,
                                                                    self.TRANSITION_STATE_OUTPUTS_DIM))

        # matrix: 13*13*2*150
        in_edge_repr = tf.identity(h)
        in_edge_repr = tf.reshape(in_edge_repr, (leng_doc, 1, 1, self.TRANSITION_STATE_OUTPUTS_DIM))
        in_edge_repr = tf.broadcast_to(in_edge_repr, (leng_doc,
                                                      leng_doc,
                                                      2,
                                                      self.TRANSITION_STATE_OUTPUTS_DIM))
        # in_edge_repr: 13*13*2*150 h(i, l)
        h_in = tf.multiply(in_edge_repr, unweight_adj_matrix)
        h_in = tf.reduce_sum(h_in, axis=(0, 2))

        out_edge_repr = tf.identity(h)
        out_edge_repr = tf.reshape(out_edge_repr, (1, leng_doc, 1, self.TRANSITION_STATE_OUTPUTS_DIM))
        out_edge_repr = tf.broadcast_to(out_edge_repr, (leng_doc,
                                                        leng_doc,
                                                        2, self.TRANSITION_STATE_OUTPUTS_DIM))
        # out_edge_repr: 13*13*2*150 h(j, l)
        h_out = tf.multiply(out_edge_repr, unweight_adj_matrix)
        h_out = tf.reduce_sum(h_out, axis=(1, 2))
        print(f'h out: {h_out}')
        return h_in, h_out


class CalculateSLayer(Layer):
    def __init__(self, graph_builder):
        super(CalculateSLayer, self).__init__()
        self.graph_builder = graph_builder
        self.dep_emb = Embedding(self.graph_builder.num_edge_type, GraphLSTM.DEP_EMB_DIM, input_length=1)
        self.dep_tanh = Dense(GraphLSTM.DEP_EMB_DIM + GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM * 2,
                              activation='tanh')

    def call(self, matrix, unpreprocessed_unweight_adj_matrix, h):
        """
        Args:
            input: a string denoting one single sentence
            h: previous hidden state
        """
        # h: 13*60
        leng_doc = matrix.shape[0]
        # matrix: 13*13*2
        # unpreprocessed_unweight_adj_matrix: 13*13*2

        unweight_adj_matrix = tf.reshape(unpreprocessed_unweight_adj_matrix, (leng_doc, leng_doc, 2, 1))
        unweight_adj_matrix = tf.broadcast_to(unweight_adj_matrix,
                                              (leng_doc,
                                               leng_doc,
                                               2,
                                               GraphLSTM.DEP_EMB_DIM + GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM * 2))
        # unweight_adj_matrix: 13*13*2*70
        embedded_matrix = self.dep_emb(matrix)
        # embedded_matrix: 13*13*2*10
        node_edge_h = tf.identity(h)
        node_edge_h = tf.reshape(node_edge_h,
                                 (leng_doc, 1, 1, 2 * GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM))
        node_edge_h = tf.broadcast_to(node_edge_h,
                                      (leng_doc, leng_doc, 2, 2 * GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM))
        node_edge_repr = tf.concat((embedded_matrix, node_edge_h), axis=3)
        node_edge_repr = tf.multiply(unweight_adj_matrix, node_edge_repr)
        # node_edge_repr: 13*13*2*70
        node_edge_repr = self.dep_tanh(node_edge_repr)
        s_in = tf.reduce_sum(node_edge_repr, axis=(0, 2))

        reversed_node_edge_h = tf.identity(h)
        reversed_node_edge_h = tf.reshape(reversed_node_edge_h,
                                          (1, leng_doc, 1, 2 * GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM))

        reversed_node_edge_h = tf.broadcast_to(reversed_node_edge_h,
                                               (leng_doc, leng_doc, 2, 2 * GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM))

        reversed_node_edge_repr = tf.concat((embedded_matrix, reversed_node_edge_h), axis=3)
        reversed_node_edge_repr = tf.multiply(unweight_adj_matrix, reversed_node_edge_repr)

        # reversed_node_edge_repr: 13*13*2*70
        reversed_node_edge_repr = self.dep_tanh(reversed_node_edge_repr)
        s_out = tf.reduce_sum(reversed_node_edge_repr, axis=(1, 2))

        return s_in, s_out


class CustomizeLSTMCell(Layer):
    TRANSITION_STATE_OUTPUTS_DIM = 150
    def __init__(self):
        super(CustomizeLSTMCell, self).__init__()
        self.have_built_weight = False

    def call(self, s_in, s_out, h_in, h_out, last_c):
        number_of_token = h_in.shape[0]
        number_units_h = h_in.shape[1]
        number_units_s = s_in.shape[1]
        if not self.have_built_weight:
            self.w_in_input = self.add_weight('w_in_input',
                                              shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.w_out_input = self.add_weight('w_out_input',
                                               shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.u_in_input = self.add_weight('u_in_input',
                                              shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.u_out_input = self.add_weight('u_out_input',
                                               shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.b_input = self.add_weight('b_input',
                                           shape=[number_units_h, ], dtype=tf.float32)
            self.w_in_output = self.add_weight('w_in_output',
                                               shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.w_out_output = self.add_weight('w_out_output',
                                                shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.u_in_output = self.add_weight('u_in_output',
                                               shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.u_out_output = self.add_weight('u_out_output',
                                                shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.b_output = self.add_weight('b_output',
                                            shape=[number_units_h, ], dtype=tf.float32)
            self.w_in_forget = self.add_weight('w_in_forget',
                                               shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.w_out_forget = self.add_weight('w_out_forget',
                                                shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.u_in_forget = self.add_weight('u_in_forget',
                                               shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.u_out_forget = self.add_weight('u_out_forget',
                                                shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.b_forget = self.add_weight('b',
                                            shape=[number_units_h, ], dtype=tf.float32)
            self.w_in_update = self.add_weight('w_in_update',
                                               shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.w_out_update = self.add_weight('w_out_update',
                                                shape=[number_units_s, number_units_h], dtype=tf.float32)
            self.u_in_update = self.add_weight('u_in_update',
                                               shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.u_out_update = self.add_weight('u_out_update',
                                                shape=[number_units_h, number_units_h], dtype=tf.float32)
            self.b_update = self.add_weight('b_update',
                                            shape=[number_units_h, ], dtype=tf.float32)
            self.have_built_weight = True
        assert number_of_token == h_out.shape[0]
        assert number_of_token == s_in.shape[0]
        assert number_of_token == s_out.shape[0]
        assert number_of_token == last_c.shape[0]
        input_gate = tf.sigmoid(tf.math.add_n((tf.matmul(s_in, self.w_in_input),
                                               tf.matmul(s_out, self.w_out_input),
                                               tf.matmul(h_in, self.u_in_input),
                                               tf.matmul(h_out, self.u_out_input))))
        output_gate = tf.sigmoid(tf.math.add_n((tf.matmul(s_in, self.w_in_input),
                                                tf.matmul(s_out, self.w_out_input),
                                                tf.matmul(h_in, self.u_in_input),
                                                tf.matmul(h_out, self.u_out_input))))
        forget_gate = tf.sigmoid(tf.math.add_n((tf.matmul(s_in, self.w_in_input),
                                                tf.matmul(s_out, self.w_out_input),
                                                tf.matmul(h_in, self.u_in_input),
                                                tf.matmul(h_out, self.u_out_input))))
        update_gate = tf.sigmoid(tf.math.add_n((tf.matmul(s_in, self.w_in_input),
                                                tf.matmul(s_out, self.w_out_input),
                                                tf.matmul(h_in, self.u_in_input),
                                                tf.matmul(h_out, self.u_out_input))))
        cell_state = tf.add(tf.multiply(forget_gate, last_c), tf.multiply(update_gate, input_gate))
        hidden_state = tf.multiply(output_gate, tf.tanh(cell_state))
        return hidden_state, cell_state


class GraphLSTM(tf.keras.Model):
    BI_LSTM_PHASE_1_OUTPUT_DIM = 30
    DEP_EMB_DIM = 10
    TRANSITION_STEP = 6

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

        self.graph_builder = AdjMatrixBuilder()
        self.dep_emb = Embedding(self.graph_builder.num_edge_type, self.DEP_EMB_DIM, input_length=1)
        self.dep_tanh = Dense(self.DEP_EMB_DIM +
                              self.BI_LSTM_PHASE_1_OUTPUT_DIM * 2,
                              activation='tanh')

        self.s_calculator = CalculateSLayer(self.graph_builder)
        self.h_calculator = CalculateHLayer()
        self.state_transition = CustomizeLSTMCell()

    def emb_single(self, doc):
        """ Returns embedding for one sentence
        Args:
            doc: one tokenized spacy doc
        Returns:
            one tensor of shape [batch_size, d1 + d2 + d3]
            d1, d2, d3 are shape of 3 types of embeddings
        """
        # get token id from vocab
        tokens = [token.text for token in doc]

        # get POS
        list_POSs = [token.pos_ for token in doc]

        # concatenate embeddings
        ps = self.pos_emb(list_POSs)
        es = self.char_emb(tokens)
        cs = self.elmo_emb(list([tokens]))
        concatenated = tf.concat([es, cs, ps], 1)

        return concatenated

    def call(self, input):
        """
        Args:
            input: a single string. TODO (after finish): infer by batch
        """
        # TODO: padding before inference?
        doc = gen_dependency_tree(input)
        matrix = self.graph_builder(doc)
        unpreprocessed_unweight_adj_matrix = self.graph_builder.get_unweighted_matrix(doc)

        emb = self.emb_single(doc)  # embedding layer

        emb = tf.expand_dims(emb, axis=0)

        emb = self.biLSTM_1(emb)

        print("emb shape: ", emb.shape)

        bi_lstm_output = tf.identity(emb)
        bi_lstm_output = tf.reshape(bi_lstm_output, (bi_lstm_output.shape[1], bi_lstm_output.shape[2]))
        s_in, s_out = self.s_calculator(matrix, unpreprocessed_unweight_adj_matrix, bi_lstm_output)
        initial_h = tf.constant(np.zeros((len(doc), CustomizeLSTMCell.TRANSITION_STATE_OUTPUTS_DIM)), dtype=tf.float32)
        initial_c = tf.constant(np.zeros((len(doc), CustomizeLSTMCell.TRANSITION_STATE_OUTPUTS_DIM)), dtype=tf.float32)
        h_history = [initial_h]
        c_history = [initial_c]
        for step in range(self.TRANSITION_STEP):
            h_in, h_out = self.h_calculator(unpreprocessed_unweight_adj_matrix, h_history[-1])
            h, c = self.state_transition(s_in, s_out, h_in, h_out, c_history[-1])
            h_history.append(h)
            c_history.append(c)

        print(h_history[-1])
        return h_history[-1]  # change this later


if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"

    data = CDRData()

    model = GraphLSTM(data)
    # model.emb_single(s)

    print(model(s))

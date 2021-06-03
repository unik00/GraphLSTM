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
        # print(f'h out: {h_out}')
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

    def add_gate_weight(self, name, shape):
        return self.add_weight(name, shape=shape, dtype=tf.float32)

    @staticmethod
    def call_gate_output(s_in, s_out, h_in, h_out, last_c, w_in, w_out, u_in, u_out, bias):
        return tf.sigmoid(tf.math.add_n((tf.matmul(s_in, w_in),
                                         tf.matmul(s_out, w_out),
                                         tf.matmul(h_in, u_in),
                                         tf.matmul(h_out, u_out),
                                         bias)))

    @staticmethod
    def broadcast_bias_weight(number_of_token, bias):
        number_of_unit = bias.shape[0]
        bias = tf.reshape(bias, (1, number_of_unit))
        bias = tf.broadcast_to(bias, (number_of_token, number_of_unit))
        return bias

    def call(self, s_in, s_out, h_in, h_out, last_c):
        number_of_token = h_in.shape[0]
        number_units_h = h_in.shape[1]
        number_units_s = s_in.shape[1]
        if not self.have_built_weight:
            self.w_in_input = self.add_gate_weight('w_in_input', [number_units_s, number_units_h])
            self.w_out_input = self.add_gate_weight('w_out_input', [number_units_s, number_units_h])
            self.u_in_input = self.add_gate_weight('u_in_input', [number_units_h, number_units_h])
            self.u_out_input = self.add_gate_weight('u_out_input', [number_units_h, number_units_h])
            self.b_input = self.broadcast_bias_weight(number_of_token,
                                                      self.add_gate_weight('b_input', [number_units_h, ]))
            self.w_in_output = self.add_gate_weight('w_in_output', [number_units_s, number_units_h])
            self.w_out_output = self.add_gate_weight('w_out_output', [number_units_s, number_units_h])
            self.u_in_output = self.add_gate_weight('u_in_output', [number_units_h, number_units_h])
            self.u_out_output = self.add_gate_weight('u_out_output', [number_units_h, number_units_h])
            self.b_output = self.broadcast_bias_weight(number_of_token,
                                                       self.add_gate_weight('b_output', [number_units_h, ]))
            self.w_in_forget = self.add_gate_weight('w_in_forget', [number_units_s, number_units_h])
            self.w_out_forget = self.add_gate_weight('w_out_forget', [number_units_s, number_units_h])
            self.u_in_forget = self.add_gate_weight('u_in_forget', [number_units_h, number_units_h])
            self.u_out_forget = self.add_gate_weight('u_out_forget', [number_units_h, number_units_h])
            self.b_forget = self.broadcast_bias_weight(number_of_token,
                                                       self.add_gate_weight('b', [number_units_h, ]))
            self.w_in_update = self.add_gate_weight('w_in_update', [number_units_s, number_units_h])
            self.w_out_update = self.add_gate_weight('w_out_update', [number_units_s, number_units_h])
            self.u_in_update = self.add_gate_weight('u_in_update', [number_units_h, number_units_h])
            self.u_out_update = self.add_gate_weight('u_out_update', [number_units_h, number_units_h])
            self.b_update = self.broadcast_bias_weight(number_of_token,
                                                       self.add_gate_weight('b_update', [number_units_h, ]))
            self.have_built_weight = True
        assert number_of_token == h_out.shape[0]
        assert number_of_token == s_in.shape[0]
        assert number_of_token == s_out.shape[0]
        assert number_of_token == last_c.shape[0]

        input_gate = self.call_gate_output(s_in, s_out, h_in, h_out, last_c,
                                           self.w_in_input, self.w_out_input,
                                           self.u_in_input, self.u_out_input,
                                           self.b_input)
        output_gate = self.call_gate_output(s_in, s_out, h_in, h_out, last_c,
                                            self.w_in_output, self.w_out_output,
                                            self.u_in_output, self.u_out_output,
                                            self.b_output)
        forget_gate = self.call_gate_output(s_in, s_out, h_in, h_out, last_c,
                                            self.w_in_forget, self.w_out_forget,
                                            self.u_in_forget, self.u_out_forget,
                                            self.b_forget)
        update_gate = self.call_gate_output(s_in, s_out, h_in, h_out, last_c,
                                            self.w_in_update, self.w_out_update,
                                            self.u_in_update, self.u_out_update,
                                            self.b_update)

        cell_state = tf.add(tf.multiply(forget_gate, last_c), tf.multiply(update_gate, input_gate))
        hidden_state = tf.multiply(output_gate, tf.tanh(cell_state))
        return hidden_state, cell_state


class EntityRelation(Layer):
    """ Given hidden node states, calculate the relation score between given Chemical and Disease"""

    REDUCED_C_DIM = 20
    REDUCED_D_DIM = 21
    POSITION_EMBEDDING_IN_DIM = 100
    POSITION_EMBEDDING_OUT_DIM = 30
    SCORE_DIM = 20

    def __init__(self):
        super().__init__()
        self.reduce_c = Dense(self.REDUCED_C_DIM, activation='tanh')
        self.reduce_d = Dense(self.REDUCED_D_DIM, activation='tanh')
        self.score = Dense(self.SCORE_DIM)
        self.position_embedding = Embedding(self.POSITION_EMBEDDING_IN_DIM,
                                            self.POSITION_EMBEDDING_OUT_DIM,
                                            input_length=1)

    def reduce(self, input_doc, list_input, h, layer):
        ret = list()
        for token_range in list_input:
            sum_over_span = tf.zeros(shape=h[0].shape)
            for token_id in range(token_range[0], token_range[1]):
                # token_id - input_doc[0].i: token.i might not be starting at zero, subtract the offset
                sum_over_span = tf.reduce_sum([sum_over_span, h[token_id - input_doc[0].i]], axis=0)
            sum_over_span = tf.expand_dims(sum_over_span, axis=0)
            ret.append((layer(sum_over_span), token_range[0]))  # TODO: ask for p_i_j elaboration
        return ret

    def call(self, node_hidden_states, input_dict):
        """
        Args:
            node_hidden_states: node hidden states returned by previous LSTM layer
            input_dict: a dict returned by emb_utils.build_data_from_file()
        """
        output = dict()
        for chemical in input_dict["Chemical"]:
            for disease in input_dict["Disease"]:
                list_c = self.reduce(input_dict['doc'],
                                     input_dict["Chemical"][chemical],
                                     node_hidden_states,
                                     self.reduce_c)
                list_d = self.reduce(input_dict['doc'],
                                     input_dict["Disease"][disease],
                                     node_hidden_states,
                                     self.reduce_d)
                a = None
                for i in range(len(list_c)):
                    for j in range(len(list_d)):
                        emb = self.position_embedding(tf.math.abs(list_c[i][1] - list_d[j][1]))
                        emb = tf.reshape(emb, [1, -1])

                        current = self.score(tf.concat((list_c[i][0], list_d[j][0], emb), axis=1))

                        if a is None:
                            a = current
                        else:
                            a = tf.math.maximum(a, current)
                output[(chemical, disease)] = a
        return output


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

        self.s_calculator = CalculateSLayer(self.graph_builder)
        self.h_calculator = CalculateHLayer()
        self.state_transition = CustomizeLSTMCell()
        self.score_layer = EntityRelation()

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

    def call(self, input_dict):
        """
        Args:
            input_dict: input dict generated by dataset class. TODO: (after finish) infer by batch
        """
        # TODO: padding before inference?
        doc = input_dict['doc']
        print(doc)

        matrix = self.graph_builder(doc)
        unpreprocessed_unweight_adj_matrix = self.graph_builder(doc, return_weighted=False)

        emb = self.emb_single(doc)  # embedding layer

        emb = tf.expand_dims(emb, axis=0)

        emb = self.biLSTM_1(emb)

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

        output = self.score_layer(h_history[-1], input_dict)
        return output


if __name__ == "__main__":
    dataset = CDRData()

    model = GraphLSTM(dataset)

    train_data = dataset.build_data_from_file(dataset.DEV_DATA_PATH, mode='intra')

    for i in range(len(train_data)):
        if len(train_data[i]['Chemical']) > 0 and len(train_data[i]['Disease']) > 0:
            print(i)
            print("output: ", model(train_data[i]))
            break

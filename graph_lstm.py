from typing import List

import tensorflow as tf
import numpy as np
from allennlp.modules.elmo import batch_to_ids, Elmo
from tensorflow.keras.layers import *

from cdr_data import CDRData, normalize
from emb_utils import *
from graph_lstm_utils import AdjListBuilder


class CNNCharEmbedding(Layer):
    PADDED_LENGTH = 100
    CONV_SIZE = 5
    NUM_FILTERS = 30
    CHAR_DIM = 30

    def __init__(self, char_dict):
        super().__init__(char_dict)

        self.char_dict = char_dict

        self.chars_emb = Embedding(len(self.char_dict) + 2,
                                   self.CHAR_DIM,
                                   input_length=self.PADDED_LENGTH,
                                   embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
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
            if c in self.char_dict:
                ret.append(self.char_dict[c])
            else:
                ret.append(len(self.char_dict) + 1)
        return ret

    def pad(self, emb_chars: List[int]):
        """Center padding one token with 0s
        Args:
            emb_chars: embedded characters of one token
        """
        front = [0] * ((self.PADDED_LENGTH - len(emb_chars)) // 2)
        end = [0] * ((self.PADDED_LENGTH - len(emb_chars) + 1) // 2)
        ret = front + emb_chars + end
        assert len(ret) == self.PADDED_LENGTH, print(len(ret), self.PADDED_LENGTH)
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
        self.POS_emb = Embedding(len(self.pos_map),
                                 self.EMB_DIM,
                                 input_length=1,
                                 embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))

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


class CustomizeLSTMCell(Layer):
    TRANSITION_STATE_OUTPUTS_DIM = 150

    def __init__(self):
        super(CustomizeLSTMCell, self).__init__()
        self.have_built_weight = False

    def add_gate_weight(self, name, shape):
        return self.add_weight(name, shape=shape, dtype=tf.float32)

    @staticmethod
    def call_gate_output(s_in, s_out, h_in, h_out, last_c, w_in, w_out, u_in, u_out, bias):
        number_of_token = s_in.shape[0]
        number_of_bias_unit = bias.shape[0]
        broadcast_bias = tf.reshape(bias, (1, number_of_bias_unit))
        broadcast_bias = tf.broadcast_to(broadcast_bias, (number_of_token, number_of_bias_unit))
        return tf.sigmoid(tf.math.add_n((tf.matmul(s_in, w_in),
                                         tf.matmul(s_out, w_out),
                                         tf.matmul(h_in, u_in),
                                         tf.matmul(h_out, u_out),
                                         broadcast_bias)))

    def call(self, s_in, s_out, h_in, h_out, last_c):
        number_of_token = h_in.shape[0]
        number_units_h = h_in.shape[1]
        number_units_s = s_in.shape[1]
        if not self.have_built_weight:
            self.w_in_input = self.add_gate_weight('w_in_input', [number_units_s, number_units_h])
            self.w_out_input = self.add_gate_weight('w_out_input', [number_units_s, number_units_h])
            self.u_in_input = self.add_gate_weight('u_in_input', [number_units_h, number_units_h])
            self.u_out_input = self.add_gate_weight('u_out_input', [number_units_h, number_units_h])
            self.b_input = self.add_gate_weight('b_input', [number_units_h, ])
            self.w_in_output = self.add_gate_weight('w_in_output', [number_units_s, number_units_h])
            self.w_out_output = self.add_gate_weight('w_out_output', [number_units_s, number_units_h])
            self.u_in_output = self.add_gate_weight('u_in_output', [number_units_h, number_units_h])
            self.u_out_output = self.add_gate_weight('u_out_output', [number_units_h, number_units_h])
            self.b_output = self.add_gate_weight('b_output', [number_units_h, ])
            self.w_in_forget = self.add_gate_weight('w_in_forget', [number_units_s, number_units_h])
            self.w_out_forget = self.add_gate_weight('w_out_forget', [number_units_s, number_units_h])
            self.u_in_forget = self.add_gate_weight('u_in_forget', [number_units_h, number_units_h])
            self.u_out_forget = self.add_gate_weight('u_out_forget', [number_units_h, number_units_h])
            self.b_forget = self.add_gate_weight('b', [number_units_h, ])
            self.w_in_update = self.add_gate_weight('w_in_update', [number_units_s, number_units_h])
            self.w_out_update = self.add_gate_weight('w_out_update', [number_units_s, number_units_h])
            self.u_in_update = self.add_gate_weight('u_in_update', [number_units_h, number_units_h])
            self.u_out_update = self.add_gate_weight('u_out_update', [number_units_h, number_units_h])
            self.b_update = self.add_gate_weight('b_update', [number_units_h, ])
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


class ScoreLayer(Layer):
    """ Given hidden node states, calculate the relation score between given Chemical and Disease"""

    REDUCED_C_DIM = 100
    REDUCED_D_DIM = 100
    POSITION_EMBEDDING_IN_DIM = 600
    POSITION_EMBEDDING_OUT_DIM = 50
    SCORE_DIM = 2

    def __init__(self):
        super().__init__()
        self.reduce_c = Dense(self.REDUCED_C_DIM, activation='tanh')
        self.reduce_d = Dense(self.REDUCED_D_DIM, activation='tanh')
        self.score = Dense(self.SCORE_DIM)
        self.position_embedding = Embedding(self.POSITION_EMBEDDING_IN_DIM,
                                            self.POSITION_EMBEDDING_OUT_DIM,
                                            input_length=1,
                                            embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))

    def reduce(self, input_doc, list_input, h, layer):
        ret = list()

        for token_range in list_input:
            sum_over_span = tf.zeros(shape=h[0].shape)
            assert token_range[1] - token_range[0] >= 0, print(input_doc)
            for token_id in range(token_range[0], token_range[1] + 1):
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

                        emb = self.position_embedding(min(abs(list_c[i][1] - list_d[j][1]),
                                                          self.POSITION_EMBEDDING_IN_DIM - 1))
                        emb = tf.reshape(emb, [1, -1])

                        current = self.score(tf.concat((list_c[i][0], list_d[j][0], emb), axis=1))

                        if a is None:
                            a = current
                        else:
                            a = tf.reduce_max((a, current), axis=0)

                output[(chemical, disease)] = tf.nn.softmax(a)
        return output


class SCalculator(Layer):
    def __init__(self):
        super(SCalculator, self).__init__()
        self.graph_builder = AdjListBuilder()
        self.edge_emb = Embedding(self.graph_builder.num_edge_type,
                                  GraphLSTM.DEP_EMB_DIM,
                                  input_length=1,
                                  embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
        self.edge_tanh = Dense(GraphLSTM.DEP_EMB_DIM + GraphLSTM. BI_LSTM_PHASE_1_OUTPUT_DIM * 2,
                               activation='tanh')
        self.output_dimension = GraphLSTM.DEP_EMB_DIM + GraphLSTM.BI_LSTM_PHASE_1_OUTPUT_DIM * 2

    def __call__(self, adj_list, bi_lstm_output):
        length_doc = bi_lstm_output.shape[0]
        s_in = [tf.zeros(self.output_dimension, dtype=tf.float32) for _ in range(length_doc)]
        s_out = [tf.zeros(self.output_dimension, dtype=tf.float32) for _ in range(length_doc)]

        for node_out in range(len(adj_list)):
            for node_in, type_of_edge in adj_list[node_out]:
                edge_type_embed = self.edge_emb(type_of_edge)

                hidden_state_node_in = bi_lstm_output[node_in, :]
                hidden_state_node_out = bi_lstm_output[node_out, :]

                edge_in_repr = tf.concat((edge_type_embed, hidden_state_node_out), axis=0)
                edge_out_repr = tf.concat((edge_type_embed, hidden_state_node_in), axis=0)

                edge_in_repr = self.edge_tanh(tf.reshape(edge_in_repr, (1, -1)))
                edge_out_repr = self.edge_tanh(tf.reshape(edge_out_repr, (1, -1)))

                edge_in_repr = tf.reshape(edge_in_repr, shape=(-1))
                edge_out_repr = tf.reshape(edge_out_repr, shape=(-1))

                s_in[node_in] += edge_in_repr
                s_out[node_out] += edge_out_repr
        s_in = tf.convert_to_tensor(s_in)
        s_out = tf.convert_to_tensor(s_out)
        return s_in, s_out


class HCalculator(Layer):
    def __init__(self):
        super(HCalculator, self).__init__()

    def __call__(self, adj_list, h):
        length_doc = h.shape[0]
        hidden_state_dim = h.shape[1]
        h_in = [tf.zeros((hidden_state_dim,), dtype=tf.float32) for _ in range(length_doc)]
        h_out = [tf.zeros((hidden_state_dim,), dtype=tf.float32) for _ in range(length_doc)]

        for node_out in range(length_doc):
            for node_in, edge_type in adj_list[node_out]:
                hidden_state_node_in = h[node_in, :]
                hidden_state_node_out = h[node_out, :]

                h_in[node_in] += hidden_state_node_out
                h_out[node_out] += hidden_state_node_in

        h_in = tf.convert_to_tensor(h_in)
        h_out = tf.convert_to_tensor(h_out)
        # print(h_in.shape)
        # print(h_out.shape)
        return h_in, h_out


class GraphLSTM(tf.keras.Model):
    def get_config(self):
        pass

    BI_LSTM_PHASE_1_OUTPUT_DIM = 150
    DEP_EMB_DIM = 10
    TRANSITION_STEP = 6
    NER_HIDDEN_LAYER_SIZE = 100
    NER_OUTPUT_LAYER_SIZE = 5

    def __init__(self, dataset, use_ner = True):
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

        self.graph_builder = AdjListBuilder()
        
        self.s_calculator = SCalculator()
        self.h_calculator = HCalculator()
        self.state_transition = CustomizeLSTMCell()
        self.score_layer = ScoreLayer()
        self.drop_out_rate = 0.2
        self.drop_out_layer = Dropout(self.drop_out_rate)
        self.use_ner = use_ner
        if use_ner:
            self.ner_hidden_layer = Dense(self.NER_HIDDEN_LAYER_SIZE, activation="tanh")
            self.ner_output_layer = Dense(self.NER_OUTPUT_LAYER_SIZE, activation="softmax")

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
        
        return self.drop_out_layer(concatenated)

    def call(self, input_dict):
        """
        Args:
            input_dict: input dict generated by dataset class. TODO: (after finish) infer by batch
        """
        # TODO: padding before inference?
        doc = input_dict['doc']

        # matrix = self.graph_builder(doc)
        # unpreprocessed_unweight_adj_matrix = self.graph_builder(doc, return_weighted=False)
        adj_list = self.graph_builder(doc)
        emb = self.emb_single(doc)  # embedding layer

        emb = tf.expand_dims(emb, axis=0)

        emb = self.biLSTM_1(emb)

        bi_lstm_output = tf.identity(emb)
        bi_lstm_output = tf.reshape(bi_lstm_output, (bi_lstm_output.shape[1], bi_lstm_output.shape[2]))
        # print(bi_lstm_output.shape)
        if self.use_ner:
            ner_hidden = self.ner_hidden_layer(self.drop_out_layer(bi_lstm_output))
            ner_output = self.ner_output_layer(ner_hidden)
            
        s_in, s_out = self.s_calculator(adj_list, bi_lstm_output)
        initial_h = tf.constant(np.zeros((len(doc), CustomizeLSTMCell.TRANSITION_STATE_OUTPUTS_DIM)), dtype=tf.float32)
        initial_c = tf.constant(np.zeros((len(doc), CustomizeLSTMCell.TRANSITION_STATE_OUTPUTS_DIM)), dtype=tf.float32)
        h_history = [initial_h]
        c_history = [initial_c]
        for step in range(self.TRANSITION_STEP):
            h_in, h_out = self.h_calculator(adj_list, h_history[-1])
            h, c = self.state_transition(s_in, s_out, h_in, h_out, c_history[-1])
            h_history.append(h)
            c_history.append(c)
            break
        node_hidden = self.drop_out_layer(h_history[-1])
        output = self.score_layer(node_hidden, input_dict)
        if self.use_ner:
            return output, ner_output
        return output


if __name__ == "__main__":
    dataset = CDRData()
    model = GraphLSTM(dataset)

    train_data = dataset.build_data_from_file(dataset.DEV_DATA_PATH)

    # '''
    # for i in range(len(train_data)):
    #     if len(train_data[i]['Chemical']) > 0 and len(train_data[i]['Disease']) > 0:
    #         print(i)
    #         print("output: ", model(train_data[i]))
    #         break
    # '''

    # e = Embedding(10,
    #           100,
    #           input_length=1)
    #           #embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
    # print(e(0))

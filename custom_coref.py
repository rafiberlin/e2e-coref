from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_hub as hub
import util

from coref_model import CorefModel



class CustomCorefIndependent(CorefModel):
    """
    Modification of Coref model in independent.py to extract span embeddings
    for all possible span (with specified max span width) in the documents.
    """
    def __init__(self, config):
        super(CustomCorefIndependent, self).__init__(config)
        self.debug = config["debug"]
        self.embeddings = self.get_idx_span(*self.input_tensors)

    def get_idx_span(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config["char_embedding_size"] > 0:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                char_emb = tf.gather(
                    tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                    char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
                char_emb = util.print_shape(char_emb, "char_emb:40", self.debug)
                flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                           util.shape(char_emb,
                                                                      3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
                flattened_char_emb = util.print_shape(flattened_char_emb, "flattened_char_emb:44", self.debug)
                flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                    "filter_size"])  # [num_sentences * max_sentence_length, emb]
                flattened_aggregated_char_emb = util.print_shape(flattened_aggregated_char_emb, "flattened_aggregated_char_emb:47", self.debug)
                aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                                 util.shape(flattened_aggregated_char_emb,
                                                                                            1)])  # [num_sentences, max_sentence_length, emb]
                aggregated_char_emb = util.print_shape(aggregated_char_emb,
                                                                 "aggregated_char_emb:51", self.debug)
                context_emb_list.append(aggregated_char_emb)
                head_emb_list.append(aggregated_char_emb)


        if not self.lm_file:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
                lm_embeddings = elmo_module(
                    inputs={"tokens": tokens, "sequence_len": text_len},
                    signature="tokens", as_dict=True)
                word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
                lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                                   lm_embeddings["lstm_outputs1"],
                                   lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]

        lm_emb = util.print_shape(lm_emb, "lm_emb:68", self.debug)
        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation", reuse=tf.AUTO_REUSE):
            self.lm_weights = tf.nn.softmax(
                tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_lm_emb = util.print_shape(flattened_lm_emb, "flattened_lm_emb:76", self.debug)
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                                 1))  # [num_sentences * max_sentence_length * emb, 1]
        flattened_aggregated_lm_emb = util.print_shape(flattened_aggregated_lm_emb, "flattened_aggregated_lm_emb:79", self.debug)
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb = util.print_shape(aggregated_lm_emb, "aggregated_lm_emb:81", self.debug)
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = util.print_shape(context_emb, "context_emb:86", self.debug)
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = util.print_shape(head_emb, "head_emb:88", self.debug)
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]
        text_len_mask = util.print_shape(text_len_mask, "text_len_mask:93", self.debug)


        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]
        context_outputs = util.print_shape(context_outputs, "context_outputs:97", self.debug)
        num_words = util.shape(context_outputs, 0)

        # genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
        #                       genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                   [1, max_sentence_length])  # [num_sentences, max_sentence_length]

        sentence_indices = util.print_shape(sentence_indices, "sentence_indices:106", self.debug)

        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_sentence_indices = util.print_shape(flattened_sentence_indices, "flattened_sentence_indices:109", self.debug)
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]
        flattened_head_emb = util.print_shape(flattened_head_emb, "flattened_head_emb:111",
                                                      self.debug)
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]

        candidate_starts = util.print_shape(candidate_starts, "candidate_starts:116",
                                                      self.debug)

        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]
        candidate_ends = util.print_shape(candidate_ends, "candidate_ends:121",
                                                      self.debug)

        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]

        candidate_start_sentence_indices = util.print_shape(candidate_start_sentence_indices, "candidate_start_sentence_indices:127",
                                                      self.debug)

        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]

        candidate_end_sentence_indices = util.print_shape(candidate_end_sentence_indices, "candidate_end_sentence_indices:133",
                                                      self.debug)

        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        candidate_mask = util.print_shape(candidate_mask, "candidate_mask:138",
                                                      self.debug)

        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]

        flattened_candidate_mask = util.print_shape(flattened_candidate_mask, "flattened_candidate_mask:143",
                                                      self.debug)

        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]

        candidate_starts = util.print_shape(candidate_starts, "candidate_starts:149",
                                                      self.debug)

        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        # candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
        #                                              flattened_candidate_mask)  # [num_candidates]
        #
        # candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
        #                                                   cluster_ids)  # [num_candidates]

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts,
                                               candidate_ends)  # [num_candidates, emb]
        # The dimension of a candidate span vector is 400 + 400 + 450 +20
        # 400 => Results of the bi-lstm applied to concatenated entry [word2vec, char cnn , elmo].
        # The first 400 is the beginning of a span, the last is the end
        # 450 is an attention mechanism applied on the vectors representating the head word. entry vector are here concat
        # [word 2 vec,char CNN]
        # The size 20 represents features of the span (the length)
        return [candidate_span_emb, candidate_starts, candidate_ends]
        # return [flattened_lm_emb, flattened_lm_emb, flattened_lm_emb]

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_start_emb = util.print_shape(span_start_emb, "span_start_emb:168", self.debug)
        span_emb_list.append(span_start_emb)
        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_end_emb = util.print_shape(span_end_emb, "span_start_emb:170", self.debug)
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                span_width_index = span_width - 1  # [k]
                span_width_emb = tf.gather(
                    tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                    span_width_index)  # [k, emb]
                span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
                span_width_emb = util.print_shape(span_width_emb, "span_width_emb:170", self.debug)
                span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                       1)  # [k, max_span_width]
            span_indices = util.print_shape(span_indices, "span_indices:189", self.debug)
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_indices = util.print_shape(span_indices, "span_indices:191", self.debug)
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            span_text_emb = util.print_shape(span_text_emb, "span_text_emb:193", self.debug)
            with tf.variable_scope("head_scores", reuse=tf.AUTO_REUSE):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_head_scores = util.print_shape(span_head_scores, "span_head_scores:197", self.debug)
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_mask = util.print_shape(span_mask, "span_mask:200", self.debug)
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_head_scores = util.print_shape(span_head_scores, "span_head_scores:202", self.debug)
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_attention = util.print_shape(span_attention, "span_attention:204", self.debug)
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_head_emb = util.print_shape(span_head_emb, "span_head_emb:206", self.debug)
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        span_emb = util.print_shape(span_emb, "span_emb:210", self.debug)
        return span_emb  # [k, emb]

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs,
                                                                                        2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)


    def restore(self, session):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        # Don't try to restore unused variables from the TF-Hub ELMo module created in the CustomCoref class.
        vars_to_restore = [v for v in vars_to_restore if "module_1/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

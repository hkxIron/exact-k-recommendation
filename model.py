# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

from layers import *
from hyperparams import Hyperparams as hp
from data_load_ml import *
from modules import *
from utils import *

# generator:1. 用transformer作为encoder
#           2. 用lstm+pointer-network作为decoder
class Generator():
    def __init__(self, is_training=True):

        # user: [batch]
        self.user = tf.placeholder(tf.int32, shape=(None,))
        # candidate_item: [batch, seq_length=candidate_item_num=50]
        self.candidate_item = tf.placeholder(tf.int32, shape=(None, hp.encoder_seq_length))
        # card_item_idx: [batch, result_length=card_item_num=10]
        self.card_item_idx = tf.placeholder(tf.int32, shape=(None, hp.result_length))

        # define decoder inputs
        # decode_target_item_idx: [batch, result_length=card_item_num=4]
        self.sampled_target_item_idx = tf.placeholder(dtype=tf.int32,
                                                      shape=[hp.batch_size, hp.result_length],  # [batch, card_item_num]
                                                      name="decoder_target_ids")  # [batch_size, res_length]
        # reward:[batch]
        self.reward = tf.placeholder(dtype=tf.float32,
                                     shape=[hp.batch_size],
                                     name="reward")  # [batch_size]

        # Load vocabulary
        user2idx, idx2user = load_user_vocab()
        item2idx, idx2item = load_item_vocab()

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            # user: [batch]
            # enc_user = [batch, hidden_units]
            self.embedding_user = embedding(self.user,
                                            vocab_size=len(user2idx),
                                            num_units=hp.hidden_units,
                                            zero_pad=False,
                                            scale=True,
                                            scope="enc_user_embed",
                                            reuse=not is_training)
            # candidate_item:[batch, seq_len=candidate_item_num]
            # enc_item = [batch, seq_len, hidden_units]
            self.embedding_item = embedding(self.candidate_item,
                                            vocab_size=len(item2idx),
                                            num_units=hp.hidden_units,
                                            zero_pad=False,
                                            scale=True,
                                            scope='enc_item_embed',
                                            reuse=not is_training)
            # embedding_user:[batch, hidden_units]
            #             => [batch, seq_len, hidden_units], 将user embed复制了多份
            # embedding_item:[batch_size, seq_len, hidden_units]
            # encoder: [batch, seq_len, 2*hidden_units]
            # 将user与item的embedding concat起来进行融合
            self.encoder = tf.concat([tf.stack(hp.encoder_seq_length * [self.embedding_user], axis=1), self.embedding_item], axis=2)

            # Dropout
            # enc: [batch, seq_len, 2*hidden_units]
            self.encoder = tf.layers.dropout(self.encoder,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

            if hp.use_multihead_attention:
                ## Blocks
                for i in range(hp.num_blocks): # 2层transformer
                    with tf.variable_scope("num_blocks_{}".format(i)): # 每个不同的层,是不同的variable_scope,即各层之间参数不共享
                        ### Multihead self Attention
                        # encoder: [batch, seq_len, 2*hidden_units]
                        #       => [batch, seq_len, 2*hidden_units]
                        self.encoder = multihead_attention(queries=self.encoder,
                                                           keys=self.encoder,
                                                           values=self.encoder,
                                                           num_units=hp.hidden_units*2,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False)

                        ### Feed Forward
                        # encoder: [batch, seq_len, 2*hidden_units]
                        self.encoder = feedforward(self.encoder, num_units=[4 * hp.hidden_units, hp.hidden_units * 2])
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=hp.hidden_units * 2)
                outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.encoder, dtype=tf.float32)
                # encoder: [batch, seq_len, 2*hidden_units]
                self.encoder = outputs

        # Decoder, decoder使用的是lstm
        with tf.variable_scope("decoder"):
            decoder_cell = LSTMCell(hp.hidden_units * 2)

            if hp.num_layers > 1:
                cells = [decoder_cell] * hp.num_layers
                decoder_cell = MultiRNNCell(cells)

            # encoder_init_sate: [batch_size, state_size=2*hidden_units]
            encoder_init_state = trainable_initial_state(hp.batch_size, decoder_cell.state_size)

            # ==============================================================================================================
            # pointer-network sampling, 从各个时间步的encoder中进行采样
            # encoder: [batch, seq_length=candidate_item_num, 2*hidden_units]
            # =>
            # sampled_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len=candidate_item_num],
            # 即可以看出是对每个样本中所有encoder output timestep计算概率分布
            # sampled_item_index: [batch_size, res_length=card_item_num=4]
            sampled_logits, sampled_item_index, _ = pointer_network_rnn_decoder(
                cell=decoder_cell,
                decoder_target_ids=None,
                encoder_outputs=self.encoder,
                enc_final_states=encoder_init_state,
                encoder_seq_length=hp.encoder_seq_length, # candidate_item_num
                result_length=hp.result_length, # card_item_num
                hidden_dim=hp.hidden_units*2,
                num_glimpse=hp.num_glimpse, # glimpse=1
                batch_size=hp.batch_size,
                mode="SAMPLE", # 多项式分布采样一个样本
                reuse=False,
                beam_size=None)

            # sampled_logits: [batch_size, res_length=card_item_num, seq_length=enoder_outputs.seq_len]
            self.sampled_logits = tf.identity(sampled_logits, name="sampled_logits")
            # sampled_item_index: [batch_size, res_length=card_item_num]
            self.sampled_item_index = tf.identity(sampled_item_index, name="sampled_item_index")

            # candidate_item: [batch, seq_length=candidate_item_num=20]
            # sampled_item_index: [batch, res_length=card_item_num=4]
            # sampled_item_seq: [batch, res_length=card_item_num=4]
            self.sampled_item_seq = batch_gather(self.candidate_item, self.sampled_item_index) # 将item index转换为真正的item_id

            # ==============================================================================================================
            # 训练得到decoder_sample_logits, 注意:这里输入的decode_target_item_idx是我们generator模型采样出来的item index,
            # 而不是我们真正ground truth的card_item_index!!!
            # self.decode_target_item_idx: [batch, result_length=card_item_num=4]
            # encoder: [batch, seq_len=cadidate_item_num, 2*hidden_units]
            # decoder_sample_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len], 即可以看出是对每个样本中所有encoder timestep计算概率分布
            decoder_sample_logits, _ = pointer_network_rnn_decoder(
                cell=decoder_cell,
                decoder_target_ids=self.sampled_target_item_idx, # [batch, result_length=card_item_num=4],
                # 从代码中看,这里输入的decode_target_item_idx是我们模型采样出来的item index,而不是我们真正ground truth的card_item_index
                encoder_outputs=self.encoder,
                enc_final_states=encoder_init_state,
                encoder_seq_length=hp.encoder_seq_length, # candidate_item_num
                result_length=hp.result_length, # card_item_num
                hidden_dim=hp.hidden_units*2,
                num_glimpse=hp.num_glimpse,
                batch_size=hp.batch_size,
                mode="TRAIN",
                reuse=True,
                beam_size=None)
            # decoder_sample_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len]
            self.decoder_sample_logits = tf.identity(decoder_sample_logits, name="dec_logits")

            # ==============================================================================================================
            # supervised_logits中的card_item_idx是真实的card item label, 不同于decoder_sample_logits中是采样出来的item
            # encoder: [batch, seq_len=cadidate_item_num, 2*hidden_units]
            # supervised_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len]
            supervised_logits, _ = pointer_network_rnn_decoder(
                cell=decoder_cell,
                decoder_target_ids=self.card_item_idx,# [batch, result_length=card_item_num=4],只有此处与上面的decoder_logits不同,是真正的card_item_index
                encoder_outputs=self.encoder,
                enc_final_states=encoder_init_state,
                encoder_seq_length=hp.encoder_seq_length, # candidate_item_num
                result_length=hp.result_length,
                hidden_dim=hp.hidden_units*2,
                num_glimpse=hp.num_glimpse,
                batch_size=hp.batch_size,
                mode="TRAIN",
                reuse=True,
                beam_size=None)
            # supervised_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len]
            self.supervised_logits = tf.identity(supervised_logits, name="supervised_logits")

            # ==============================================================================================================
            # beamsearch推断网络,即预测target_id_index,搜索一个最好的item序列
            # infer_card_item_idx: [batch_size, res_length=card_item_num=4]
            _, infer_card_item_idx, _ = pointer_network_rnn_decoder(
                cell=decoder_cell,
                decoder_target_ids=None,
                encoder_outputs=self.encoder,
                enc_final_states=encoder_init_state,
                encoder_seq_length=hp.encoder_seq_length,
                result_length=hp.result_length,
                hidden_dim=hp.hidden_units*2,
                num_glimpse=hp.num_glimpse,
                batch_size=hp.batch_size,
                mode="BEAMSEARCH",
                reuse=True,
                beam_size=hp.beam_size)

            # infer_card_item_idx: [batch_size, res_length = card_item_num = 4]
            self.infer_card_item_idx = tf.identity(infer_card_item_idx, name="infer_card_item_idx")
            # candidate_item:      [batch, seq_length=candidate_item_num=20]
            # infer_card_item_idx: [batch, res_length=card_item_num=4]
            # infer_card_item:     [batch, res_length=card_item_num=4]
            self.infer_card_item = batch_gather(self.candidate_item, self.infer_card_item_idx)


        if is_training:
            # Loss
            # decoder_sample_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len]
            # decode_target_item_idx: [batch, result_length=card_item_num=4]
            # reinforcement_loss: [batch, result_length], 这个loss能够称为强化学习的loss,因为是采样来的吗?
            self.reinforcement_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_sample_logits,  # 对于sparse_softmax函数,labels不用one-hot
                                                                                     labels=self.sampled_target_item_idx)

            # decoder_sample_logits: [batch_size, res_length=card_item_num=4, seq_length=enoder_outputs.seq_len]
            # card_item_idx:   [batch, result_length=card_item_num=4]
            # supervised_loss: [batch, result_length]
            if hp.schedule_sampling:
                self.supervised_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_sample_logits,
                                                                                      labels=self.card_item_idx) # 真实的有监督的card item
            else:
                self.supervised_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.supervised_logits,
                                                                                      labels=self.card_item_idx)

            # reinforcement
            # reinforcement_loss: [batch,result_length=card_item_index]
            # reward: [batch]
            # policy_loss:scalar
            # sum(loss)*reward,后面我们将看到,reward来源于discriminator的reward,各个时间序列上的loss相加
            # TODO:不明白为何reward乘以loss就是policy_loss
            self.policy_loss = tf.reduce_mean(tf.reduce_sum(self.reinforcement_loss, axis=1) * self.reward)

            # supervised_loss: [batch, result_length]
            # =>: scalar
            self.supervised_loss = tf.reduce_mean(tf.reduce_sum(self.supervised_loss, axis=1))
            # 强化学习的loss+有监督学习的loss相加
            # policy_loss:scalar
            # supervised_loss: scalar
            self.loss = (1.0 - hp.supervised_coe) * self.policy_loss + hp.supervised_coe * self.supervised_loss

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr_generator,
                                                    beta1=0.9,
                                                    beta2=0.98,
                                                    epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step)

# 判别器
class Discriminator():
    def __init__(self, is_training=True, is_testing=False):
        if is_training and is_testing:
            raise TypeError('is_training and is_testing cannot be both true!')

        # user:[batch]
        # card_item:[batch, card_item_num=4]
        # label:[batch]
        # num_batch:scalar
        if is_training: #训练
            # 此处user,card_item等都已经是tensor
            self.user, self.card_item, self.label, self.num_batch = get_discriminator_batch_data(is_training=True)
        elif is_testing: # 测试
            self.user, self.card_item, self.label, self.num_batch = get_discriminator_batch_data(is_training=False)
        else: # 预测时的 user以及card_item是临时输入的,而不是从文件中读取的
            self.user = tf.placeholder(tf.int32, shape=(hp.batch_size,))
            self.card_item = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.result_length))

        # Load vocabulary
        user2idx, idx2user = load_user_vocab()
        item2idx, idx2item = load_item_vocab()

        ## Embedding
        # encoder_user_embedding = [batch_size, hidden_units]
        self.encoder_user_embedding = embedding(self.user,
                                                vocab_size=len(user2idx),
                                                num_units=hp.hidden_units,
                                                zero_pad=False,
                                                scale=True,
                                                scope="enc_user_embed",
                                                reuse= not is_training)

        # encoder_card_item_embedding = [batch_size, res_len=card_item_num=4, hidden_units]
        self.encoder_card_item_embedding = embedding(self.card_item,
                                                     vocab_size=len(item2idx),
                                                     num_units=hp.hidden_units,
                                                     zero_pad=False,
                                                     scale=True,
                                                     scope='enc_card_embed',
                                                     reuse=not is_training)

        ## Dropout
        # encoder_user_embedding = [batch_size, hidden_units]
        self.encoder_user_embedding = tf.layers.dropout(self.encoder_user_embedding,
                                                        rate=hp.dropout_rate,
                                                        training=tf.convert_to_tensor(is_training))

        # encoder_card_item_embedding = [batch_size, res_len=card_item_num=10, hidden_units]
        self.encoder_card_item_embedding = tf.layers.dropout(self.encoder_card_item_embedding,
                                                             rate=hp.dropout_rate,
                                                             training=tf.convert_to_tensor(is_training))
        # discriminator_logits:[batch]
        self.discriminator_logits = ctr_dicriminator(user_embedding=self.encoder_user_embedding,
                                                     card_item_embedding=self.encoder_card_item_embedding,
                                                     hidden_dim=hp.discriminator_hidden_size)
        # discriminator_probs:[batch]
        self.discriminator_probs = tf.sigmoid(self.discriminator_logits)

        # 由于随机猜对的概率是0.5,因此以0.5作为baseline, 即只有>0.5才会有正的 reward,小于则是负的reward
        # discriminator_probs:[batch]
        self.discriminator_reward = (self.discriminator_probs - 0.5) * 2.0

        if is_training or is_testing:
            # self.label:[batch]
            # discriminator_logits:[batch]
            # discriminator_loss: scalar
            self.discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                                             logits=self.discriminator_logits))
            # accurancy scalar:概率是否>0.5
            self.discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater_equal(self.discriminator_probs, 0.5)),
                                                                              self.label)))

        if is_training:
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr_discriminator, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.discriminator_loss, global_step=self.global_step)

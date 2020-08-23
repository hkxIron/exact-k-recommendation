# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

from layers import *
from hyperparams import Hyperparams as hp
from data_load_ml import *
from modules import *
import os, codecs
from tqdm import tqdm
from utils import *
from model import Generator, Discriminator

if __name__ == '__main__':
    # load generator data, 生成器的数据
    generator_user, generator_card_item, generator_card_idx, generator_candidate_item, generator_pos_item, generator_num_batch \
        = get_generator_batch_data(is_training=True)
    generator_user_test, generator_card_item_test, _, generator_candidate_item_test, generator_pos_item_test, generator_num_batch_test \
        = get_generator_batch_data(is_training=False)

    # Construct graph
    with tf.name_scope('Generator'):
        generator = Generator(is_training=True)
    print("generator variable count:", len(tf.get_variable_scope().global_variables()))

    with tf.name_scope('Discriminator'):
        discriminator = Discriminator(is_training=True, is_testing=False)
    print("disriminator variable count:", len(tf.get_variable_scope().global_variables()))

    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('DiscriminatorInfer'):
        discriminator_infer = Discriminator(is_training=False, is_testing=False)

    with tf.name_scope('DiscriminatorTest'):
        discriminator_test = Discriminator(is_training=False, is_testing=True)

    # 每个在不同的命名空间中
    with tf.name_scope('GeneratorInfer'):
        generator_infer = Generator(is_training=False)

    print("Graph loaded")
    # Load vocabulary
    user2idx, idx2user = load_user_vocab()
    item2idx, idx2item = load_item_vocab()

    # log file init
    # generator
    gen_train_log = open(os.path.join(hp.logdir, hp.gen_train_log_path), 'w')
    gen_train_log.write('step\tgen_reward\tprecision@4\tprecision\n')
    gen_test_log = open(os.path.join(hp.logdir, hp.gen_test_log_path), 'w')
    gen_test_log.write('step\tgen_reward\tprecision@4\tprecision\n')
    # discriminator
    dis_train_log = open(os.path.join(hp.logdir, hp.dis_train_log_path), 'w')
    dis_train_log.write('step\tdis_loss\tdis_acc\n')
    dis_test_log = open(os.path.join(hp.logdir, hp.dis_test_log_path), 'w')
    dis_test_log.write('step\tdis_loss\tdis_acc\n')

    # Start session
    sv = tf.train.Supervisor(is_chief= True,
                             summary_op=None,
                             logdir=hp.logdir,
                             save_model_secs=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)  # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with sv.managed_session(config=sess_config) as sess:
        print('Discriminator training start!')

        discriminator_accurancy_best = 0.0
        discriminator_loss_total, discriminator_accurancy_total = 0.0, 0.0
        for epoch in range(1, hp.discriminator_num_epochs + 1): # discriminator训练一次
            if sv.should_stop():
                break
            print('Discriminator epoch: ', epoch)

            # for step in tqdm(range(d.num_batch), total=d.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(discriminator.num_batch):
                global_step_discriminator = sess.run(discriminator.global_step)
                _, discriminator_loss, discriminator_accurancy = sess.run([discriminator.train_op, discriminator.discriminator_loss, discriminator.discriminator_accuracy])
                discriminator_loss_total += discriminator_loss
                discriminator_accurancy_total += discriminator_accurancy

                ## print
                if (global_step_discriminator + 1) % hp.print_per_step == 0:
                    print('global_step_discriminator: {}, discriminator_loss_train: {}, discriminator_acc_train: {}'.format(
                        (global_step_discriminator + 1),
                        discriminator_loss_total / (1.0 * (global_step_discriminator + 1)),
                        discriminator_accurancy_total / (1.0 * (global_step_discriminator + 1))))
                    dis_train_log.write('{}\t{}\t{}\n'.format(
                        (global_step_discriminator + 1),
                        discriminator_loss_total / (1.0 * (global_step_discriminator + 1)),
                        discriminator_accurancy_total / (1.0 * (global_step_discriminator + 1))))
                    dis_train_log.flush()

                ## test
                if (global_step_discriminator + 1) % hp.test_per_step == 0:
                    discriminator_loss_test, discriminator_accurancy_test = 0.0, 0.0
                    for _ in range(discriminator_test.num_batch):
                        discriminator_loss, discriminator_accurancy = sess.run([discriminator_test.discriminator_loss, discriminator_test.discriminator_accuracy])
                        discriminator_loss_test += discriminator_loss
                        discriminator_accurancy_test += discriminator_accurancy

                    discriminator_loss_test /= (1.0 * discriminator_test.num_batch)
                    discriminator_accurancy_test /= (1.0 * discriminator_test.num_batch)
                    print('global_step_discriminator: {}, discriminator_loss_test: {}, discriminator_accurancy_test: {}'.format(
                        (global_step_discriminator + 1), discriminator_loss_test, discriminator_accurancy_test))
                    dis_test_log.write('{}\t{}\t{}\n'.format((global_step_discriminator + 1), discriminator_loss_test, discriminator_accurancy_test))
                    dis_test_log.flush()

                    # 将最好的discriminator保存成文件
                    if discriminator_accurancy_test > discriminator_accurancy_best:
                        discriminator_accurancy_best = discriminator_accurancy_test
                        print('discriminator_accurancy_best: ', discriminator_accurancy_best)
                        sv.saver.save(sess, hp.logdir + '/model/best_model')

        print('Discriminator training done!')

        # 恢复最好的discriminator
        sv.saver.restore(sess, hp.logdir + '/model/best_model')

        print('Generator training start!')
        # 记录sample到的最好的结果
        memory_reward = {}
        memory_card_item_idx = {}
        memory_card_item = {}

        precision_at_4_best, precision_best = 0.0, 0.0
        reward_total, precision_at_4_total, precision_total = 0.0, 0.0, 0.0
        for epoch in range(1, hp.generator_num_epochs + 1):
            if sv.should_stop():
                break
            print('Generator epoch: ', epoch)

            # for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(generator_num_batch):
                user, card_item, card_item_idx, candidate_item, pos_item = \
                    sess.run([generator_user, generator_card_item, generator_card_idx, generator_candidate_item, generator_pos_item])

                if hp.is_hill_climbing:
                    samples = []
                    for i in range(hp.batch_size):
                        # user:[batch]
                        # user[i]:1
                        # user_i:[num_hill_climb]
                        # candidate_item:[batch, candidate_item_num=20]
                        # candidate_item[i]: [1, candidate_item_num=20]
                        # item_cand_i: [num_hill_climb, candidate_item_num=20]
                        user_i = np.tile(user[i], reps=(hp.num_hill_climb)) # 每个user生成num_hill_climb个样本,同时进行探索
                        item_cand_i = np.tile(candidate_item[i], reps=(hp.num_hill_climb, 1))
                        # 用generator生成序列
                        # sample_path: [batch_size, res_length]
                        hill_sampled_card_item_idx, hill_sampled_card_item = sess.run([generator.sampled_item_index, generator.sampled_item_seq],
                                                                                      feed_dict={generator.user: user_i,
                                                                                                 generator.candidate_item: item_cand_i})
                        # 用discriminator推断reward
                        hill_reward = sess.run(discriminator_infer.discriminator_reward,
                                               feed_dict={discriminator_infer.card_item: hill_sampled_card_item,
                                                          discriminator_infer.user: user_i})
                        # 按reward进行降序排序
                        sorted_list = sorted(list(zip(hill_sampled_card_item, hill_sampled_card_item_idx, hill_reward)),
                                             key=lambda x: x[2], reverse=True)
                        samples.append(sorted_list[np.random.choice(hp.top_k_candidate)]) #从list中随机选取3个item, top_k_candidate=3

                        # 以前这个user的reward未被记录
                        if user[i] not in memory_reward:
                            memory_reward[user[i]] = sorted_list[0][2]
                            memory_card_item_idx[user[i]] = sorted_list[0][1]
                            memory_card_item[user[i]] = sorted_list[0][0]
                        else: # 该user已被记录,更新该user的reward为最大的reward
                            if memory_reward[user[i]] > sorted_list[0][2]:
                                memory_reward[user[i]] = sorted_list[0][2]
                                memory_card_item_idx[user[i]] = sorted_list[0][1]
                                memory_card_item[user[i]] = sorted_list[0][0]
                    # 将采样的batch个样本以及reward返回
                    (sampled_card_item, sampled_card_item_idx, reward) = zip(*samples)
                else:
                    # sample
                    # 用generator生成序列, 与爬山法不同, 每个user只采样一次
                    sampled_card_item_idx, sampled_card_item = sess.run([generator.sampled_item_index, generator.sampled_item_seq],
                                                                        feed_dict={generator.user: user,
                                                                                   generator.candidate_item: candidate_item})

                    if hp.use_discriminator_reward: # 使用判别器的reward
                        reward = sess.run(discriminator_infer.discriminator_reward,
                                          feed_dict={discriminator_infer.card_item: sampled_card_item,
                                                     discriminator_infer.user: user})
                    else: # 生成器采样card item的reward
                        reward = []
                        for i in range(len(sampled_card_item)):
                            if pos_item[i] in set(sampled_card_item[i]):
                                reward.append(1.0) # 如果包含正样本,就认为reward是1.0
                            else:
                                reward.append(-1.0)

                # train generator
                sess.run(generator.train_op, feed_dict={generator.decode_target_item_idx: sampled_card_item_idx,
                                                        generator.reward: reward,
                                                        generator.candidate_item: candidate_item,
                                                        generator.user: user,
                                                        generator.card_item_idx: card_item_idx})
                global_step_generator = sess.run(generator.global_step)
                reward_total += np.mean(reward)

                # beamsearch
                beamsearch_card_item = sess.run(generator_infer.infer_card_item,
                                                feed_dict={generator_infer.candidate_item: candidate_item,
                                                           generator_infer.user: user})
                precision_at_4_total += precision_at_4(beamsearch_card_item, pos_item)
                precision_total += precision(beamsearch_card_item, card_item)

                ## print
                if (global_step_generator + 1) % hp.print_per_step == 0:
                    print('global_step_generator: {}, generator_reward_train: {}, precision@4_train: {}, precision_train: {}'.format(
                        (global_step_generator + 1),
                        reward_total / (1.0 * (global_step_generator + 1)),
                        precision_at_4_total / (1.0 * (global_step_generator + 1)),
                        precision_total / (1.0 * (global_step_generator + 1))))
                    gen_train_log.write('{}\t{}\t{}\t{}\n'.format(
                        (global_step_generator + 1),
                        reward_total / (1.0 * (global_step_generator + 1)),
                        precision_at_4_total / (1.0 * (global_step_generator + 1)),
                        precision_total / (1.0 * (global_step_generator + 1))))
                    gen_train_log.flush()

                ## test
                if (global_step_generator + 1) % hp.test_per_step == 0:
                    precision_at_4_test, precision_test, reward_test = 0.0, 0.0, 0.0
                    for _ in range(generator_num_batch_test):
                        user_test, card_item_test, candidate_item_test, pos_item_test \
                            = sess.run([generator_user_test, generator_card_item_test,
                                        generator_candidate_item_test, generator_pos_item_test])
                        beamsearch_card_item_test = sess.run(generator_infer.infer_card_item,
                                                             feed_dict={generator_infer.candidate_item: candidate_item_test,
                                                                        generator_infer.user: user_test})
                        precision_at_4_test += precision_at_4(beamsearch_card_item_test, pos_item_test)
                        precision_test += precision(beamsearch_card_item_test, card_item_test)
                        reward = sess.run(discriminator_infer.discriminator_reward,
                                          feed_dict={discriminator_infer.card_item: beamsearch_card_item_test,
                                                     discriminator_infer.user: user_test})
                        reward_test += np.mean(reward)

                    reward_test /= (1.0 * generator_num_batch_test)
                    precision_at_4_test /= (1.0 * generator_num_batch_test)
                    precision_test /= (1.0 * generator_num_batch_test)
                    print('global_step_generator: {}, generator_reward_test: {}, precision@4_test: {}, precision_test: {}'.format(
                        (global_step_generator + 1), reward_test, precision_at_4_test, precision_test))
                    gen_test_log.write('{}\t{}\t{}\t{}\n'.format(
                        (global_step_generator + 1), reward_test, precision_at_4_test, precision_test))
                    gen_test_log.flush()

                    if precision_at_4_test > precision_at_4_best:
                        precision_at_4_best = precision_at_4_test
                        precision_best = precision_test
                        print('precision_at_4_best: ', precision_at_4_best,
                              'precision_best: ', precision_best)
                        sv.saver.save(sess, hp.logdir + '/model/best_model')

        print('Generator training done!')

    print("Done")

    gen_train_log.close()
    gen_test_log.close()
    dis_train_log.close()
    dis_test_log.close()
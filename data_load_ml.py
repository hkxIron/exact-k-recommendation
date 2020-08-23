# coding:utf-8
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import random

def load_user_vocab():
    user_ids = [line.strip() for line in open(hp.user_ids_file, 'r').read().splitlines()]
    user2idx = {int(user): idx for idx, user in enumerate(user_ids)}
    idx2user = {idx: int(user) for idx, user in enumerate(user_ids)}
    return user2idx, idx2user

def load_item_vocab():
    item_ids = [line.strip() for line in open(hp.item_ids_file, 'r').read().splitlines()]
    item2idx = {int(item): idx for idx, item in enumerate(item_ids)}
    idx2item = {idx: int(item) for idx, item in enumerate(item_ids)}
    return item2idx, idx2item

##########################################
# generator是从candidate item中选取部分item作为card item
def load_generator_data(file_path):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    # USER:[batch]
    # CARD_ITEM:[batch, card_item_num=4]
    # CARD_ITEM_IDX:[batch, card_item_num=4]
    # CANDIDATE_ITEM:[batch, candidate_item_num=20]
    # ITEM_POS:[batch]
    USER, CARD_ITEM, CARD_ITEM_IDX, CANDIDATE_ITEM, POS_ITEM = [], [], [], [], []
    # 文件数据格式:user_id card_item_ids candidate_item_ids
    with open(file_path, 'r') as fin:
        for line in fin:
            strs = line.strip().split('\t')
            USER.append(user2idx[int(strs[0])])
            # 注意:card_item一定在candidate item集合里
            card_items = [item2idx[int(x)] for x in strs[1].split(',')]
            CARD_ITEM.append(card_items)
            # 作者应该也是将点击率最高的item放在第一位
            POS_ITEM.append(card_items[0]) # 最后点击该card中的哪个item,默认将点击的item排在第一位,如果没有点击,label就是0

            candidate_items = sorted([item2idx[int(x)] for x in strs[2].split(',')]) # candidate item
            CANDIDATE_ITEM.append(candidate_items) # sorted

            item_candidate_to_idx_map = {}
            for idx, candidate_item in enumerate(candidate_items):
                item_candidate_to_idx_map[candidate_item] = idx
            # 计算每个展示card中的item在candidate list中的index
            card_item_index = [item_candidate_to_idx_map[item] for item in card_items]
            CARD_ITEM_IDX.append(card_item_index)

            '''
            tmp = set(strs[2].split(','))
            tmp.remove(strs[1].split(',')[0])
            tmp = list(tmp)
            random.shuffle(tmp)
            ITEM_CAND_NEG.append([item2idx[int(x)] for x in tmp])
            '''
    return USER, CARD_ITEM, CARD_ITEM_IDX, CANDIDATE_ITEM, POS_ITEM

def get_generator_batch_data(is_training=True):
    # Load data
    if is_training:
        USER, CARD_ITME, CARD_ITEM_IDX, CANDIDATE_ITEM, POS_ITEM = load_generator_data(hp.gen_data_train)
        batch_size = hp.batch_size
        print('Load generator training data done!')
    else: # 测试集
        USER, CARD_ITME, CARD_ITEM_IDX, CANDIDATE_ITEM, POS_ITEM = load_generator_data(hp.gen_data_test)
        batch_size = hp.batch_size
        print('Load generator testing data done!')

    # calc total batch count
    num_batch = len(USER) // batch_size

    # Convert to tensor
    USER = tf.convert_to_tensor(USER, tf.int32) # [batch_size]
    CARD_ITME = tf.convert_to_tensor(CARD_ITME, tf.int32) # [batch_size, card_item_num=4], 每个card只展示4个item
    CARD_ITEM_IDX = tf.convert_to_tensor(CARD_ITEM_IDX, tf.int32) # [batch_size, card_item_num=4]
    CANDIDATE_ITEM = tf.convert_to_tensor(CANDIDATE_ITEM, tf.int32) # [batch_size, candidate_time_num=20]
    POS_ITEM = tf.convert_to_tensor(POS_ITEM, tf.int32) # [batch_size],正样本,应该是需要预测的正样本
    # ITEM_CAND_NEG = tf.convert_to_tensor(ITEM_CAND_NEG, tf.int32) # [batch_size, 19]

    # Create Queues
    input_queues = tf.train.slice_input_producer([USER, CARD_ITME, CARD_ITEM_IDX, CANDIDATE_ITEM, POS_ITEM])

    # create batch queues
    user, card_item, card_item_idx, candidate_item, pos_item = \
        tf.train.shuffle_batch(input_queues,
                               num_threads=8,
                               batch_size=batch_size,
                               capacity=batch_size * 64,
                               min_after_dequeue=batch_size * 32,
                               allow_smaller_final_batch=False)
    # card_neg = tf.random_crop(item_cand_neg, size=[hp.batch_size, hp.res_length])
    return user, card_item, card_item_idx, candidate_item, pos_item, num_batch

#####################################
# discriminator通过预测用户是否点击
def load_discriminator_data(file_path):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    USER, CARD_ITEMS, LABEL = [], [], []
    # 格式:user,item list,label
    # 683	511,911,754,286,914,358,344,312,683,347	1
    with open(file_path, 'r') as fin:
        for line in fin:
            strs = line.strip().split('\t')
            USER.append(user2idx[int(strs[0])])
            card_items = [item2idx[int(x)] for x in strs[1].split(',')] # 每个卡片里有多个商品,我们来预测这个卡片要不要被点击
            random.shuffle(card_items)  # shuffled
            CARD_ITEMS.append(card_items) # 获取卡片中所有的item
            LABEL.append(float(strs[2])) # 卡片整体会不会被点击

    return USER, CARD_ITEMS, LABEL

# 判别器的数据都是有label,即是有监督的
def get_discriminator_batch_data(is_training=True):
    # Load data
    if is_training:
        USER, CARD_ITEMS, LABEL = load_discriminator_data(hp.discriminator_data_train)
        batch_size = hp.batch_size
        print('Load discriminator training data done!')
    else:
        USER, CARD_ITEMS, LABEL = load_discriminator_data(hp.discriminator_data_test)
        batch_size = hp.batch_size
        print('Load discriminator testing data done!')

    # calc total batch count
    num_batch = len(USER) // batch_size

    # Convert to tensor
    USER = tf.convert_to_tensor(USER, tf.int32)  # [batch_size]
    CARD_ITEMS = tf.convert_to_tensor(CARD_ITEMS, tf.int32)  # [batch_size, 4]
    LABEL = tf.convert_to_tensor(LABEL, tf.float32)  # [batch_size]

    # Create Queues
    input_queues = tf.train.slice_input_producer([USER, CARD_ITEMS, LABEL])

    # create batch queues
    user, card_item, label = \
        tf.train.shuffle_batch(input_queues,
                               num_threads=8,
                               batch_size=batch_size,
                               capacity=batch_size * 64,
                               min_after_dequeue=batch_size * 32,
                               allow_smaller_final_batch=False)
    # user:[batch]
    # card_item:[batch, card_item_num=4]
    # label:[batch]
    # num_batch:scalar
    return user, card_item, label, num_batch

if __name__ == "__main__":
    user, card_item, card_item_idx, candidate_item, pos_item, num_batch = get_generator_batch_data(is_training=True)
    print(user)
    print(card_item)
    print(card_item_idx)
    print(candidate_item)
    print(pos_item)
    print(str(num_batch))

    user, card_item, label, num_batch = get_discriminator_batch_data(is_training=True)
    print(user)
    print(card_item)
    print(label)
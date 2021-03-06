# encoding: UTF-8
import tensorflow as tf

# input:
# a:[[3,1,2],
#    [2,3,1]]-> c:[[[0, 3], [0, 1], [0, 2]], # 0代表第0个样本, 即 c[i,j,:] = [i,a[i,j]]
#                  [[1, 2], [1, 3], [1, 1]]] # 1代表第1个样本
#
# input:
# [3,1,4] => [[0,3], # [sample_index, seq_index]
#             [1,1],
#             [2,4]]
def index_matrix_to_pairs_fn(batch_size, seq_length):
    replicated_first_indices = tf.range(batch_size)  # range(128)
    # replicated_first_indices =
    #    [[  0,  0,  0,...],
    #     [  1,  1,  1,...],
    #     ......
    #     [127,127,127,...]]
    replicated_first_indices2 = tf.tile(
        input=tf.expand_dims(replicated_first_indices, dim=1),  # [128,1]
        multiples=[1, seq_length])

    def index_matrix_to_pairs(index_matrix):
        """
        :param index_matrix: [batch_size, data_len] or [batch_size]即为a
        :return: [batch_size, data_len, 2] or [batch_size, 2],即为c
        ie:
          a: [128, 10] -> c[i,j,:] = [i,a[i,j]], shape(c) = [128,10,2]
          a: [128] -> c[i,:] = [i,a[i]], shape(c) = [128,2]
        """
        rank = len(index_matrix.get_shape())
        if rank == 1:
            return tf.stack([replicated_first_indices, index_matrix], axis=rank)
        elif rank == 2:
            return tf.stack([replicated_first_indices2, index_matrix], axis=rank)
        else:
            raise NotImplementedError("index_matrix rank should be 1 or 2, but %d found" % rank)

    return index_matrix_to_pairs


"""
data: [[3 4 5]
       [7 8 9]]
indices: [[0 2]# 选取下标第0,2的元素
          [1 0]]# 选取下标第1,0的元素
out: [[3 5] 
      [8 7]] 
"""
def batch_gather(data, indices):
    """
    一般情况下，data的shape为[batch_size, T], indices的shape为[batch_size, res_length]
    :param data: 需要从中选择的数据
    :param indices: 需要选择数据的index
    :param batch_size: batch_size
    :param gather_data_size: 每批要选出的数据条数
    :return:
    """
    batch_size = data.get_shape()[0].merge_with(indices.get_shape()[0]).value
    if batch_size is None:
        batch_size = tf.shape(indices)[0]
    gather_data_size = indices.get_shape()[1].value
    if gather_data_size is None:
        gather_data_size = tf.shape(indices)[1]
    flat_indices = tf.reshape(tf.transpose(indices), (-1,))  #[batch*4,1]
    input_index_pairs = tf.stop_gradient(tf.stack(
             [tf.range(batch_size*gather_data_size, dtype=tf.int32), flat_indices], axis=1))
    flat_data = tf.tile(data, [gather_data_size, 1])
    return tf.transpose(tf.reshape(tf.gather_nd(flat_data, input_index_pairs), (gather_data_size, batch_size)))

# card里的item个数为4
def precision_at_4(card_infer, item_pos):
    # card_item_infer: [batch, res_length=card_item_num]
    # item_pos:      [batch]
    acc_sum = 0.0
    for sample_index in range(len(card_infer)):
        if item_pos[sample_index] in set(card_infer[sample_index]):
            acc_sum += 1.0
    return acc_sum / (1.0 * len(card_infer))

def precision(card_infer, card_label):
    # card_item_infer: [batch, res_length=card_item_num]
    # card_label:      [batch, res_length=card_item_num]
    acc_sum = 0.0
    for sample_index in range(len(card_infer)):
        tmp = 0.0
        for x in card_infer[sample_index]:
            if x in set(card_label[sample_index]): #当前item在card_label中
                tmp += 1.0 # 分数+1
        acc_sum += (tmp / (1.0 * len(card_infer[sample_index]))) # 当前样本预测的item中,有几个是在label中,算一个准确率
    return acc_sum / (1.0 * len(card_infer)) # 所有样本的平均准确率


if __name__ == '__main__':
    print("run test")
    data =tf.constant([[3, 4, 5],
                       [7, 8, 9]])
    indices = tf.constant([[0, 2],
                           [1, 0]])
    res = batch_gather(data, indices)
    with tf.Session() as sess:
        out = sess.run(res)
        print("data:", sess.run(data))
        print("indices:",sess.run(indices))
        print("out:", out)
        """
        data: [[3 4 5]
         [7 8 9]]
        indices: [[0 2]
         [1 0]]
        out: [[3 5]
         [8 7]]
        """


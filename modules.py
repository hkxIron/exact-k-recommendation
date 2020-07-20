# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf
import numpy as np

def layer_normalization(inputs,
                        epsilon = 1e-8,
                        scope="ln",
                        reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # inputs: [N, T_q, d_model].
        inputs_shape = inputs.get_shape() # (N, T_q, d_model)
        params_shape = inputs_shape[-1:] # 取出最后的维度,(d_model,)

        # inputs: [N, T_q, d_model].
        # mean: [N, T_q, 1],只在最后一个维度上进行求平均
        # variance: [N, T_q, 1],只在最后一个维度上进行求方差
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # beta与gamma是需要学习的参数
        # beta:[d_model,]
        beta= tf.Variable(tf.zeros(params_shape))
        # gamma:[d_model,]
        gamma = tf.Variable(tf.ones(params_shape))
        # inputs: [N, T_q, d_model].
        # mean: [N, T_q, 1]
        # normalized: [N, T_q, d_model].
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) ) # (x-mu)/sigma
        """
        注意:此处的gamma在各个维度上的值并不相同,即各维度上不共享
        """
        # gamma:[d_model,]
        # normalized: [N, T_q, d_model].
        # beta:[d_model,]
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.random_uniform_initializer(-0.08, 0.08))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs



def multihead_attention(queries, 
                        keys,
                        values,
                        num_units=None,
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]. 即[batch, seq_len, channel_dim]
      keys: A 3d tensor with shape of [N, T_k, C_k].
      values: A 3d tensor with shape of [N, T_v=T_k, C_v]. 在attention里keys与value的seq长度是一样的,即T_v = T_k
      num_units: A scalar. Attention size. 在attention空间里的维度,下面表示为 C
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. mask一些未来的unit
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C = num_units)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1] # [N, T_q, C_q]最后一维: C=C_q
        
        # Linear projections, 将Q, K, V 先转换到attention空间
        # queries:[N, T_q, C_q]
        # Q:[N, T_q, C]
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # keys:[N, T_k, C_k]
        # K:[N, T_k, C]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu) # (N, T_v=T_k, C)
        
        # Split and concat
        # Q:[N, T_q, C]
        # =>[h*N, T_q, C/h], h = num_heads, 将最后一维分成多头head
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        # K_: (h*N, T_k, C/h)
        # => [h*N, C/h, T_k]
        # Q:[h*N, T_q, C/h]
        # outputs:[h*N, T_q, T_k], 含义是Q的每个时间步对K的每个时间步进行交互
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        attention_dim = K_.get_shape().as_list()[-1] # C/h
        outputs = outputs / (attention_dim ** 0.5) # 缩放

        """
        key mask的意图就是为了将<pad>的部分的embedding置为很小的负数, embedding=0对应的是index=0,即<pad>
        同理,query mask也是将<pad>部分置为很小的负数
        """
        # Key Masking
        # keys:[N, T_k, C_k]
        #   => [N, T_k, C_k]
        #key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k), transformer的代码里是先abs再reduce_sum,然后sign
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k), 对batch里的每个序列的各维度embedding相加并求和取绝对值然后取sign
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k), 为每个头复制一份mask
        # 为queries复制T_q份
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        # key_masks:[h*N, T_q, T_k], key_mask等于0的地方用padding, 否则用outputs
        # paddings: [h*N, T_q, T_k]
        # outputs:  [h*N, T_q, T_k]
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            """
            生成inputs的关于时间的掩码下三角矩阵

            此处是因果推断,即预测T时刻时不能提前看到T时刻的标签
            input: [[-1.4095545  -0.5366828  -0.5652379 ]
                    [ 0.526246   -0.11131065  0.26350743]]
            tril: [[-1.4095545   0.          0.        ]
                   [ 0.526246   -0.11131065  0.        ]] 
            """
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k) , LinearOperatorLowerTriangular
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1) # 一个绝对值很大的负数
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs, axis=-1) # (h*N, T_q, T_k), 每个query对各个keys的attention值进行归一化
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, T_k)
          
        # Dropouts
        # outputs:(h*N, T_q, T_k)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        # outputs:(h*N, T_q, T_k)
        # V_:(h*N, T_k, C/h)
        # outputs:(h*N, T_q, C/h)
        outputs = tf.matmul(outputs, V_) # (h*N, T_q, C/h)
        
        # Restore shape
        # outputs:(h*N, T_q, C/h), h = num_heads
        #      => (N, T_q, C)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = layer_normalization(outputs) # (N, T_q, C)
 
    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    每个位置各自前向传播,各个位置之间并不发生交互
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        # inputs:[N, T, C]
        params = {"inputs": inputs,
                  "filters": num_units[0], # output_channel,即filter的个数
                  "kernel_size": 1,
                  "stride":1,
                  "padding":'valid',
                  "activation": tf.nn.relu, # 用relu激活
                  "use_bias": True
                  }
        # conv1d是先将tensor升维后经过conv2d处理, 然后再降维
        # inputs: [batch=N, max_enc_length=T, input_channel=C]
        #       => [batch, in_height=1, in_width=seq_length, in_channels]
        # filter_kernel: [filter_width=1, input_channel=C, output_channel=num_units[0]]
        #             => [filter_height=1, filter_width=1, input_channels=C, output_channels=num_units[0]]
        # conv2d的维度大小: out_size = (img_size+2*pad-filter_size)//stride+1 = (img_size-1)//1+1=img_size,即保持原来大小
        # outputs: [batch, out_height=1, out_width=max_enc_length=T, output_channel=num_units[0]]
        #                  => [batch=N, seq_length=max_enc_length=T, output_channel=num_units[0]]
        # 理解:可以看出来,作者选用的是1*1的卷积,即基于单像素在不同通道上的卷积
        outputs = tf.layers.conv1d(**params) # 此处用卷积代替全连接
        
        # Readout layer
        params = {"inputs": outputs,
                  "filters": num_units[1],
                  "kernel_size": 1,
                  "stride":1,
                  "padding":'valid',
                  "activation": None,
                  "use_bias": True
                  }
        # 下面的卷积同理
        # outputs: [N, T, num_units[1]]
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        # inputs: [N, T, C]
        # outputs: [N, T, num_units[1]=C], 此处要求 num_units[1]=C
        outputs += inputs
        
        # Normalize
        # outputs: [N, T, C]
        outputs = layer_normalization(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
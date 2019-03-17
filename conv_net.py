import tensorflow as tf

def conv_net(image,batch_size):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                shape=[3,3,3,16],  #卷积核大小，输出个数
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[16],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0,1))
        conv = tf.nn.conv2d(image,weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation, name='conv1')
    
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling1')
        pool1_lrn = tf.nn.lrn(pool1,depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pooling1_lrn')
    
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                shape=[3,3,16,128],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[128],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0,1))
        conv = tf.nn.conv2d(pool1_lrn, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2_lrn = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pooling2_lrn')
        pool2 = tf.nn.max_pool(pool2_lrn, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME', name='pooling2')
    
    with tf.variable_scope('full_con1') as scope:
        flatten = tf.reshape(pool2, shape=[batch_size,-1])
        dim = flatten.get_shape()[1].value
        weights = tf.get_variable('weights',
                                shape=[dim,4096],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[4096],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0,1))
        full_con1 = tf.nn.relu(tf.matmul(flatten, weights)+biases, name='full_con1')
    
    with tf.variable_scope('full_con2') as scope:
        weights = tf.get_variable('weights',
                                shape=[4096,2048],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[2048],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0,1))
        full_con2 = tf.nn.relu(tf.matmul(full_con1, weights)+biases, name='full_con2')
    
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                shape=[2048,3],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                shape=[3],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0,1))
        softmax_linear = tf.add(tf.matmul(full_con2,weights),biases, name='softmax_linear')
    return softmax_linear
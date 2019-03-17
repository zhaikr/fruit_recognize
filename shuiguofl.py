import os
import tensorflow as tf
import keras
from skimage import io,transform
import skimage
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle


##改变当前路径到数据路径
os.chdir('/home/zhaikr/fruits-360')
'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True 
'''
##读取图片和标签
def load_small_data(dir_path,m):
    images_m = []
    labels_m = []
    lab = os.listdir(dir_path)
    n = 0
    for l in lab:
        if(n>=m):
            break
        img = os.listdir(dir_path+l)
        for i in img:
            img_path = dir_path+l+'/'+i
            labels_m.append(l)
            #print(n)
            images_m.append(img_path)
        n+=1
    temp = np.array([images_m,labels_m])
    temp = temp.transpose()
    np.random.shuffle(temp)  ##数据打乱
    images_list = list(temp[:,0])
    labels_list = list(temp[:,1])
    labels_list = [int(i) for i in labels_list]
    return images_list,labels_list

##image存储的是路径，需要读取为图片
def get_batches(image,label,resize_h,resize_w,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels = 3)
    image = tf.image.resize_image_with_crop_or_pad(image,resize_h,resize_w)  ##图片resize处理为大小一致
    image = tf.image.per_image_standardization(image)  ##即减去所有图片的均值，方便训练s

    image_batch,label_batch = tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads=64, #使用多个线程在tensor_list中读取文件
                                            capacity=capacity)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])
    return images_batch,labels_batch
    
def conv_net(image,batch_size):
    ## Convolutional layer 1(卷积层1)
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
    
    ## Max pooling(池化，向下采样层)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling1')
        pool1_lrn = tf.nn.lrn(pool1,depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pooling1_lrn')
    
    ## Convolutional layer 2（卷积层2）
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
    
     ## Max pooling(池化，向下采样层)
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2_lrn = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pooling2_lrn')
        pool2 = tf.nn.max_pool(pool2_lrn, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME', name='pooling2')
    
    ## Fully connected layer 1(全连接层1)
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
    
    ## Fully connected layer 2(全连接层2)
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
    
    ## Output class prediction
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


##定义评估量
##定义损失函数，这是用于训练最小化损失的必需量
def loss(logits,labels_batches):
    cross_entory = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_batches)
    cost = tf.reduce_mean(cross_entory)
    return cost

##评价分类准确率的量，训练时，需要loss值减小，准确率增加，这样的训练才是收敛的。
def get_accuracy(logits,labels):
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc

##定义训练方式
def optimizer(loss, num):
    train_op = tf.train.AdadeltaOptimizer(num).minimize(loss)
    return train_op

def run_training():
    pic_path = './Training/'  ##数据存放的路径
    image,label = load_small_data(pic_path,3)
    #print(label)
    image_batches,label_batches = get_batches(image,label,32,32,3,20)
    predication = conv_net(image_batches,3)
    cost = loss(predication,label_batches)
    train_op = optimizer(cost,0.0009) ##学习率为0.0009
    acc = get_accuracy(predication,label_batches) ##准确率

    sess = tf.Session()  ##会话开始
    init = tf.global_variables_initializer()  ##参数初始化
    sess.run(init)
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(500000):  ##训练次数
            if coord.should_stop():
                break
            _,train_acc,train_loss = sess.run([train_op,acc,cost])
            print("iteration", '%04d'%(step+1), "cost", '{:.9f}'.format(train_loss), "Train accuracy", "{:.5f}".format(train_acc))
            if step%100000 == 0:  ##多少次保存一次模型
                model = '/home/zhaikr/shuiguo/model/model'  ##模型保存位置
                saver.save(sess,model,global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done!!!')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

run_training()







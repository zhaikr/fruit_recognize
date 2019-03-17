import tensorflow as tf
import matplotlib.pyplot as plt
import conv_net
import numpy as np
from PIL import Image

##加载图片
def get_one_image(image_path):
    image = Image.open(image_path)
    image = image.resize([32,32])
    image_arr = np.array(image)
    return image_arr

##对图片进行分类
def test(test_file):
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,32,32,3])
        p = conv_net.conv_net(image,1)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[32,32,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./model/')
            if ckpt and ckpt.model_checkpoint_path:
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
            prediction =sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            print(prediction,max_index)
    return max_index

fruit = test('/home/zhaikr/shuiguo/timg2.jpeg')
fruit_dict = {0:'apple', 1:'banana'}  ##标签对应的字典
print(fruit_dict[fruit])
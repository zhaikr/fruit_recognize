import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import np_utils
from sklearn.utils import shuffle
import tensorflow as tf

##改变当前路径
os.chdir('/home/zhaikr/fruits-360')
#np.set_printoptions(threshold=np.inf) 

'''
##获取对应路径下的图片
def load_data(dir_path):
    images = []  ##列表存放图片
    labels = []  ##存放标签
    no5_imgs = []  ##每种图片的第五张图片
    label_no5 = []  ##第五张图片的标签
    lab = os.listdir(dir_path)  ##浏览文件夹
    n = 0
    for l in lab:
        img = os.listdir(dir_path+l)
        for i in img:
            img_path = dir_path+l+'/'+i
            labels.append(int(n))
            #print(n)
            images.append(skimage.data.imread(img_path))
            #print(img_path)
        n+=1
        no5_img = format_path(img)
        img5_path = dir_path+l+'/'+no5_imgcorrect_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        label_no5.append(l)
        no5_imgs.append(skimage.data.imreacorrect_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return images,labels,no5_imgs,label_nocorrect_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##按顺序读取文件内图片的函数
def format_path(img):
    yes_int = []
    for s in range(len(img)):
        img[s] = img[s].split('_') ##以_将correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        if(is_number(img[s][0])):
            img[s][0] = int(img[s][0])
            yes_int.append(img[s])
    yes_int.sort()
    for yi in range(len(yes_int)):
        yes_int[yi][0] = str(yes_int[yi][0correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        yes_int[yi] = yes_int[yi][0]+'_'+ycorrect_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    no5_img = yes_int[4]
    return no5_img

##判断数据是否可以转换成整形
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (ValueError,TypeError):
        pass
    
    return False
'''

'''
##最终数据集与标签
images,labels,no5_imgs,labels_no5=load_data('./Training/')
##print(len(images),len(labels),len(no5_imgs))
'''

'''
##最终测试数据集与标签
images_test,labels_test,no5_imgs_test=load_data('./Test/')
print(len(images_test),len(labels_test),len(no5_imgs_test),len(labels_no5)
'''

'''
##第五张图片的可视化
def display_no5_img(no5_imgs,label_no5):
    fig = plt.figure(figsize = (15,15))
    for i in range(len(no5_imgs)):
        plt.subplot(11,9,(i+1))ImportError: libcublas.so.10.0: cannot open shared object file: No such file
        plt.title(format(label_no5[i]))
        plt.imshow(no5_imgs[i])
        plt.axis('off')
    plt.show()
'''

'''
##调用显示函数
display_no5_img(no5_imgs,labels_no5) 
'''

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
            labels_m.append(int(l))
            #print(n)
            images_m.append(skimage.data.imread(img_path))
        n+=1
    return images_m,labels_m

'''
##批量剪裁图片
def cut_image(images,w,h):
    new_image = [skimage.transform.resize(I,(w,h)) for I in images]
    return new_image
'''
##预处理数据打乱数据同时独热编码
def prepare_data(images,labels,n_classes):
    #print(labels)
    train_x = np.array(images)
    train_y = np.array(labels)
    #print(train_y)
    index = np.arange(0,train_y.shape[0])
    index = shuffle(index)
    #print(index)
    train_x = train_x[index]
    #print(train_x)
    train_y = train_y[index]
    train_y = keras.utils.to_categorical(train_y,n_classes) ##one-hot独热编码
    #print(train_y)
    return train_x,train_y

images_20,labels_20=load_small_data('./Training/',4) ##训练集
images_test_20,labels_test_20=load_small_data('./Test/',4) ##测试集

##训练集数据预处理
train_x,train_y=prepare_data(images_20,labels_20,4)
##测试数据集与标签的数组化和乱序
test_x,test_y=prepare_data(images_test_20,labels_test_20,4)


## 配置神经网络的参数
n_classes = 4  ##数据的类别数
batch_size = 128  ##训练块的大小
kernel_h = kernel_w = 5  ##卷积核尺寸
dropout = 0.8  ##dropout的概率
depth_in = 3  ##图片的通道数
depth_out1 = 64  ##第一层卷积的卷积核个数
depth_out2 = 128  ##第二层卷积的卷积核个数
image_size = train_x.shape[1]   ##图片尺寸
n_sample = train_x.shape[0]  ##训练样本个数
t_sample = test_x.shape[0]   ##测试样本个数

##feed给神经网络的图像数据类型与shape,四维，第一维训练的数据量，第二、三维图片尺寸，第四维图像通道数
x = tf.placeholder(tf.float32,[None,100,100,3],name='Mul')
y = tf.placeholder(tf.float32,[None,n_classes],name='y_') ##feed到神经网络的标签数据的类型和shape

keep_prob = tf.placeholder(tf.float32)  ##dropout的placeholder(解决过拟合)
fla = int((image_size*image_size/16)*depth_out2)  ##用于扁平化处理的参数经过两层卷积池化后的图像大小*第二层的卷积核个数

##定义各卷积层和全连接层的权重变量
Weight = {"con1_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_in,depth_out1])),\
        "con2_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_out1,depth_out2])),\
        "fc_w1":tf.Variable(tf.random_normal([int((image_size*image_size/16)*depth_out2),1024])),\
        "fc_w2":tf.Variable(tf.random_normal([1024,512])),\
        "out":tf.Variable(tf.random_normal([512,n_classes]))}


##定义各卷积层和全连接层的偏置变量
bias = {"conv1_b":tf.Variable(tf.random_normal([depth_out1])),\
        "conv2_b":tf.Variable(tf.random_normal([depth_out2])),\
        "fc_b1":tf.Variable(tf.random_normal([1024])),\
        "fc_b2":tf.Variable(tf.random_normal([512])),\
        "out":tf.Variable(tf.random_normal([n_classes]))}


## 定义卷积层的生成函数
def conv2d(x,W,b,stride=1):
    x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

## 定义池化层的生成函数
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")

## 定义卷积神经网络生成函数
def conv_net(x,weight,bias,dropout):
    
    ## Convolutional layer 1(卷积层1)
    conv1 = conv2d(x,Weight['con1_w'],bias['conv1_b'])
    conv1 = maxpool2d(conv1,2)

    ## Convolutional layer 2（卷积层2）
    conv2 = conv2d(conv1,Weight['con2_w'],bias['conv2_b'])
    conv2 = maxpool2d(conv2,2)
    
    ## Fully connected layer 1(全连接层1)
    flatten = tf.reshape(conv2,[-1,fla])  ##Flatten层，扁平化处理
    fc1 = tf.add(tf.matmul(flatten,Weight['fc_w1']),bias['fc_b1'])
    fc1 = tf.nn.relu(fc1)  ##经过relu激活函数
    print(flatten.get_shape())
    
    ## Fully connected layer 2(全连接层2)
    fc2 = tf.add(tf.matmul(fc1,Weight['fc_w2']),bias['fc_b2'])  ##计算公式：输出参数=输入参数*权值+偏置
    fc2 = tf.nn.relu(fc2) ##经过relu激活函数
    
    ## Dropout（Dropout层防止预测数据过拟合）
    fc2_drop = tf.nn.dropout(fc2,dropout)
    
    ## Output class prediction
    prediction = tf.add(tf.matmul(fc2_drop,Weight['out']),bias['out'],name='final_result')  ##输出预测参数
    return prediction




prediction=conv_net(x,Weight,bias,keep_prob) ##生成卷积神经网络
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y)) ##交叉熵损失函数
optimizer=tf.train.AdamOptimizer(0.0009).minimize(cross_entropy)


## 评估模型
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))



##训练块数据生成器将一次训练的数据分成n块小数据，feed给神经网络
def gen_small_data(inputs,batch_size):
    i = 0
    while True:
        small_data = inputs[i:(batch_size+i)]
        i += batch_size
        yield small_data


saver = tf.train.Saver()  ##模型保存
## 初始会话并开始训练过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(4):
        train_x,train_y = prepare_data(images_20,labels_20,4)  ##重新预处理数据
        train_x = gen_small_data(train_x,50)  ##生成图像块数据
        train_y = gen_small_data(train_y,50)  ##生成标签块数据
        for j in range(int(n_sample/batch_size+1)):
            x_=next(train_x)
            y_=next(train_y)
            ##准备验证数据
            validata_feed = {x:x_,y:y_,keep_prob:0.8}
            if i % 1 == 0:
                sess.run(optimizer,feed_dict=validata_feed)
                loss,acc =sess.run([cross_entropy,accuracy],feed_dict=validata_feed)
                saver.save(sess,'/home/zhaikr/shuiguo/model/model')  ##模型保存位置
                print("Epoch", '%04d'%(i+1), "cost", '{:.9f}'.format(loss), "Train accuracy", "{:.5f}".format(acc))
    
    print("optimization completed")

    ##准备测试数据
    test_x = test_x[0:400]
    test_feed = {x:test_x}
    y1 = sess.run(prediction,feed_dict=test_feed)
    test_classes = np.argmax(y1,1)
    print("Testing accuracy:",sess.run(accuracy,feed_dict=test_feed))



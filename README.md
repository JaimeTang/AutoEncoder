Tensorflow是Google推出的深入学习框架，也是目前最为流行的开源框架之一，使用Tensorflow能够完成很多图片处理的任务。
在本文中将使用图片加解码的原理，然后配合一个简单的卷积神经网络，训练出一个能够实现打了码赛克的图片复现的功能。
卷积神经网络是深度学习中训练神经网络最常用的一种模型，对图片的处理有很好的效果。文中说到的图片加解码流程，简单的理解就是对图片进行压缩和解压过程，而在卷积神经网络中的体现可以看做是进行卷积使原始图片降低维度，然后再做一次逆过程使降维后的图片升维。
下文展示的是部分实现代码
首先是导入Tensorflow 的库函数
import tensorflow as tf
命名为tf基本上是行业规范。之后是导入MNIST的数据集，文中进行训练的数据集是MNIST数据集，MNIST是一个手写开源共享的数字数据库,它有60000个训练样本集和10000个测试样本集。
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
如果本地没有存储数据集，执行这条语句后系统会自动进行下载，存放在'MNIST_data'文件夹内。

查看图片的shape，发现维度是(784,)，其实导入的MNIST数据集里面所有的图片维度都是(784,)，而且图片还是单通道的，所以如果要做可视化的时候还要进行一次shape转换。
img = mnist.train.images[20]
img = img.reshape((28,28))
plt.imshow(img)
不过还有个更加简单的方法，直接导入数据的时候就让图片的shape就是（28,28），reshape设置为False。
mnist = input_data.read_data_sets('MNIST_data', one_hot = True, reshape = False)
之后是代码的核心部分，首先定义的是卷积神经网络的的输入和输出，因为开始的输入和最终的输出都是一张28*28*1的图片，所以这里的占位变量为 (None, 28, 28, 1)，None等待的是batch的输入。
x = tf.placeholder(tf.float32, (None, 28, 28, 1), name = 'input')
y = tf.placeholder(tf.float32,(None, 28, 28, 1), name = 'output')
定义卷积神经网络中每层的权重和偏至，这里用字典的形式保存每层卷积网络的权重值和偏至值，之后就可以方便调用了。权重的初始值使用的是缩减正态分布随机生成的（这里标准差设置为0.1,这样使权重集中在负0.1和0.1的范围之内，没有这个参数后面训练cost会达到10w以上，耗费更多的训练时间），因为权重使用的是缩减的正态分布，所以偏至可以全部初始化为0。
weights = {
    'conv1':tf.Variable(tf.truncated_normal([3, 3, 1, 64],stddev = 0.1)),
    'conv2':tf.Variable(tf.truncated_normal([3, 3, 64, 64],stddev = 0.1)),
    'conv3':tf.Variable(tf.truncated_normal([3, 3, 64, 32],stddev = 0.1)),
    'conv4':tf.Variable(tf.truncated_normal([3, 3, 32, 32],stddev = 0.1)),
    'conv5':tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev = 0.1)),
    'conv6':tf.Variable(tf.truncated_normal([3, 3, 64, 64],stddev = 0.1)),
    'output':tf.Variable(tf.truncated_normal([3, 3, 64, 1],stddev = 0.1))
}

biases = {
    'b1':tf.Variable(tf.zeros([64])),
    'b2':tf.Variable(tf.zeros([64])),
    'b3':tf.Variable(tf.zeros([32])),
    'b4':tf.Variable(tf.zeros([32])),
    'b5':tf.Variable(tf.zeros([64])),
    'b6':tf.Variable(tf.zeros([64])),
    'b_output':tf.Variable(tf.zeros([1]))
}
定义卷积神经网络模型结构，这里conv2d、maxpooling2d和upsampling是自己定义的函数名称（可在完整代码文档查看），这里重点描述的是卷积神经网络结构，可以看出是3层卷积+3层池化+3层Upsampling+3层逆卷积+1层输出卷积的网络模型架构。这一系列的过程就是上文说的先降维然后再升维的过程。
conv1 = conv2d(x, weights['conv1'], biases['b1'])
conv1 = maxpooling2d(conv1)

conv2 = conv2d(conv1, weights['conv2'], biases['b2'])
conv2 = maxpooling2d(conv2)

conv3 = conv2d(conv2, weights['conv3'], biases['b3'])
conv3 = maxpooling2d(conv3)

conv4 = upsampling(conv3, 7, 7)
conv4 = conv2d(conv4, weights['conv4'], biases['b4'])

conv5 = upsampling(conv4, 14,14)
conv5 = conv2d(conv5, weights['conv5'], biases['b5'])

conv6 = upsampling(conv5, 28,28)
conv6 = conv2d(conv6, weights['conv6'], biases['b6'])

output = tf.nn.conv2d(conv6, weights['output'], strides = [1,1,1,1], padding ='SAME')
output = tf.nn.bias_add(output, biases['b_output'])
定义卷积神经网络模型训练时的Loss和优化函数，虽然代码比较简单，但是这个是卷积神经网络的核心，因为一切机器学习的目的就是一个找最优解的过程。这里激活函数使用的是sigmoid，因为我们做的不是一个分类问题，所有不必选择softmax。优化算法使用的是Adma。
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)



生成码赛克图片的代码如下，方法就是添加随机像素然后再进行一次修剪，修剪的目的是为了达到原始图片的0到1的像素分布值。
noisy_imgs = imgs + 0.5*np.random.randn(*imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
然后选好epochs（训练批次）和batch_size（没次训练使用的数据集大小）就可以让模型跑起来了。这里选择2个epoch，实践发现两个已经能够达到很好的效果了，而且会有过拟合的趋势。
epochs = 2
batch_size = 128

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    for i in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))
        noisy_imgs = imgs + 0.5*np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        batch_cost, _ = sess.run([cost,optimizer], feed_dict={x:noisy_imgs, y:imgs})
        if i % 10 == 0:
            print ('Epoch: {}/{} Batch: {}'.format(epoch+1, epochs, i),'Trianing loss: {:.4f}'.format(batch_cost))
训练完成后loss已经降到了0.1左右了。

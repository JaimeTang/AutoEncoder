1 什么是Tensorflow

Tensorflow是Google推出的深入学习框架，也是目前最为流行的开源框架之一。
2、什么是AutoEncoder

在本文中将使用AutoEncoder的原理配合一个简单的卷积神经网络，训练出一个能够实现马赛克图片复现的模型。

AutoEncoder简单的理解就是图片压缩和解压，而在卷积网络中的体现可以看做是进行卷积后然后再做一次逆过程。

下面开始展示实现代码

首先是

import tensorflow as tf

导入的是Tensorflow 的库函数，tf命名基本上是行业规范了。

之后是导入MNIST的数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

如果没有数据集执行这条语句后会进行自动下载，存放在'MNIST_data'文件夹内。

导入完成数据之后可以做一下简单的数据集探索，比如显示一张图片


查看图片的shape，发现MNIST数据集里面说有的图片都是(784,)，表示图片是单通道的，做可视化的时候还要做一次shape转换

img = mnist.train.images[20]
img = img.reshape((28,28))
plt.imshow(img)

如果想直接导入数据的时候让图片的shape就是（28,28），数据导入可以写成

mnist = input_data.read_data_sets('MNIST_data', one_hot = True, reshape = False)

之后是重要的部分，定义卷积的输入和输出

x = tf.placeholder(tf.float32, (None, 28, 28, 1), name = 'input')
y = tf.placeholder(tf.float32,(None, 28, 28, 1), name = 'output')

定义卷积网络的权重和偏至，用字典的形式保存每层卷积网络的权重值和偏至值

weights = {
'conv1':tf.Variable(tf.random_normal([5, 5, 1, 64])),
'conv2':tf.Variable(tf.random_normal([5, 5, 64, 64])),
'conv3':tf.Variable(tf.random_normal([5, 5, 64, 32])),
'conv4':tf.Variable(tf.random_normal([5, 5, 32, 32])),
'conv5':tf.Variable(tf.random_normal([5, 5, 32, 64])),
'conv6':tf.Variable(tf.random_normal([5, 5, 64, 64])),
'output':tf.Variable(tf.random_normal([5, 5, 64, 1]))
}

biases = {
'b1':tf.Variable(tf.random_normal([64])),
'b2':tf.Variable(tf.random_normal([64])),
'b3':tf.Variable(tf.random_normal([32])),
'b4':tf.Variable(tf.random_normal([32])),
'b5':tf.Variable(tf.random_normal([64])),
'b6':tf.Variable(tf.random_normal([64])),
'b_output':tf.Variable(tf.random_normal([1]))
}

之后就是定义卷积网络模型

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

output = conv2d(conv6, weights['output'], biases['b_output'])

然后定义卷积网络模型训练时的Loss和优化函数

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

最后就是进行模型的训练

epochs = 2
batch_size = 200

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            imgs = batch[0].reshape((-1, 28, 28, 1))
            noisy_imgs = imgs + 0.5*np.random.randn(*imgs.shape)
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            batch_cost, _ = sess.run([cost,optimizer], feed_dict={x:noisy_imgs, y:imgs})

            print ('Epoch: {}/{}'.format(epoch+1, epochs),'Trianing loss: {:.4f}'.format(batch_cost))
然后我们抽取用10个测试集内部的图片来验证训练结果的可靠性
完整代码实现：



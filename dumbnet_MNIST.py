import tensorflow as tf
import numpy as np
from helper import MNIST
from matplotlib import pyplot as plt
from layers import feed_forward_layer, conv_layer

def get_siamnese(X):
    """
    creates the siamnese stem for the neural network. This is the part both images are send through sequentially
    :param X: the input tensor
    :return:
    """
    with tf.variable_scope('Layer1'):
        conv1 = conv_layer(X,16,5,5,activation_function=tf.nn.relu)

    with tf.variable_scope('Layer2'):
        conv2 = conv_layer(conv1,32,3,3,activation_function=tf.nn.relu)

    with tf.variable_scope('Layer4'):
       conv3 = conv_layer(conv2,128,2,2,activation_function=tf.nn.relu)

    flat = tf.layers.flatten(conv3)
    return flat

# Define the Network Graph
with tf.variable_scope('Input'):
    X1 = tf.placeholder(tf.float32,shape=[None,28,28,1],name='Img1')
    #X2 = tf.placeholder(tf.float32,shape=[None,28,28,1],name='Img2')
    Y = tf.placeholder(tf.float32, shape=[None,10],name='Labels')

with tf.variable_scope('Siamese_Segment') as scope:
    left_net = get_siamnese(X1)
    #scope.reuse_variables()
    #right_net = get_siamnese(X2)

with tf.variable_scope('Dense_Segment'):
    concatenated = left_net
    d1 = feed_forward_layer(concatenated,1024,activation_function=tf.nn.relu)#dense(concatenated,4096*2,4096,tf.nn.relu,'Dense1')

# logits have their own scope to avoid variable dublication
with tf.variable_scope('Logit'):
    logits = feed_forward_layer(d1,10)


with tf.variable_scope('Loss_Metrics_and_Training'):
    # use weighted loss, since images of two different classes statically outnumber the matches by 10:1
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))

    # calculate the accuracy
    predicted_class = tf.argmax(logits,1)
    ground_truth = tf.argmax(Y,1)
    correct = tf.equal(predicted_class,ground_truth)
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )

    # use GradientDecent to train, interestingly ADAM results in a collapsing model. Standard SGD performed reliably better
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# set up parameters
batch_size = 256
half_batch = batch_size // 2
epochs = 10

# boring houskeeping
loss_list = []
acc_list = []
l = 0.0
init = tf.global_variables_initializer()

def get_one_hot(labels):
    oh_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        oh_labels[i][labels[i]] = 1
    return oh_labels

with tf.Session() as sess:
    sess.run(init)
    v_loss = []
    v_acc = []
    for epoch in range(epochs):
        m = MNIST('.')
        print('[INFO] Epoch:',epoch,l)
        i = 0
        for data, labels in m.get_training_batch(batch_size):
            # reshape data to fit the tf-standard shapes for images
            data = data.reshape((data.shape[0],28,28,1))
            labels = get_one_hot(labels)


            # run training step
            _,l = sess.run([train_step,loss],feed_dict={X1:data,Y:labels})
            loss_list.append(l)
            # test model on validation data
            if i%100 == 0:
                v_ll = []
                v_ac = []

                for data, labels in m.get_validation_batch(batch_size):

                    # same code as in the training loop
                    data = data.reshape((data.shape[0],28,28,1))
                    labels = get_one_hot(labels)

                    pc,l,a = sess.run([predicted_class,loss,accuracy],feed_dict={X1:data,Y:labels})

                    # keep track of loss and accuracy
                    v_ll.append(l)
                    v_ac.append(a)
            v_loss.append([np.mean(v_ll)]*100)
            v_acc.append([np.mean(v_ac)]*100)
            i += 1


    # plot the loss curve.
    plt.plot(list(range(len(loss_list))),loss_list, label='Training Loss')
    plt.plot(list(range(len(loss_list))),loss_list, label='Validation Loss')
    plt.legend()
    plt.show()

    # print the mena accuracy over the test-set
    print('Val Acc: ',np.mean(v_acc))

        # test model on validation data
    loss_list = []
    acc_list = []
    l = 0.0
    for data, labels in m.get_test_batch(batch_size):

        # same code as in the training loop
        data = data.reshape((data.shape[0],28,28,1))
        labels = get_one_hot(labels)

        pc,l,a = sess.run([predicted_class,loss,accuracy],feed_dict={X1:data,Y:labels})

        # keep track of loss and accuracy
        loss_list.append(l)
        acc_list.append(a)

    # print the mena accuracy over the test-set
    print('Test Acc:',np.mean(acc_list))

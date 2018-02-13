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
    X2 = tf.placeholder(tf.float32,shape=[None,28,28,1],name='Img2')
    Y = tf.placeholder(tf.float32, shape=[None,1],name='Labels')

with tf.variable_scope('Siamese_Segment') as scope:
    left_net = get_siamnese(X1)
    scope.reuse_variables()
    right_net = get_siamnese(X2)

with tf.variable_scope('Dense_Segment'):
    concatenated = tf.concat([left_net,right_net],axis=1,name='concatenated_out')
    d1 = feed_forward_layer(concatenated,1024,activation_function=tf.nn.relu)#dense(concatenated,4096*2,4096,tf.nn.relu,'Dense1')

# logits have their own scope to avoid variable dublication
with tf.variable_scope('Logit'):
    logits = feed_forward_layer(d1,1)


with tf.variable_scope('Loss_Metrics_and_Training'):
    # use weighted loss, since images of two different classes statically outnumber the matches by 10:1
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(Y,logits,0.1,'Loss'))

    # calculate the accuracy
    predicted_class = tf.greater(tf.nn.sigmoid(logits),0.5)
    correct = tf.equal(predicted_class, tf.equal(Y,1.0))
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )

    # use GradientDecent to train, interestingly ADAM results in a collapsing model. Standard SGD performed reliably better
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# set up parameters
batch_size = 256
half_batch = batch_size // 2
epochs = 100

# boring houskeeping
loss_list = []
acc_list = []
l = 0.0
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        m = MNIST('.')
        print('[INFO] Epoch:',epoch,l)
        for data, labels in m.get_training_batch(batch_size):
            # reshape data to fit the tf-standard shapes for images
            data = data.reshape((data.shape[0],28,28,1))

            # use a batch of data from the Data Provider and reshape it to suit the needs of the siamnese network
            # the first half of the batch is matched with the second half and the labels are calculated accordingly
            x1 = data[:half_batch]
            x2 = data[half_batch:]
            y = np.zeros((half_batch,1))
            y1 = labels[:half_batch]
            y2 = labels[half_batch:]

            # calculate binary labels for the siamnese network
            y[y1 == y2] = 0.0
            y[y1 != y2] = 1.0

            # run training step
            _,l = sess.run([train_step,loss],feed_dict={X1:x1,X2:x2,Y:y})
            loss_list.append(l)

    # test model on validation data
    for data, labels in m.get_validation_batch(batch_size):

        # same code as in the training loop
        data = data.reshape((data.shape[0],28,28,1))
        x1 = data[:half_batch]
        x2 = data[half_batch:]
        y = np.zeros((half_batch,1))
        y1 = labels[:half_batch]
        y2 = labels[half_batch:]
        y[y1 == y2] = 0.0
        y[y1 != y2] = 1.0
        pc,l,a = sess.run([predicted_class,loss,accuracy],feed_dict={X1:x1,X2:x2,Y:y})

        # keep track of loss and accuracy
        loss_list.append(l)
        acc_list.append(a)

    # print the mena accuracy over the test-set
    print('Acc:',np.mean(acc_list))

    # plot the labels of all pairs in the last batch that assumed to have the same label
    # only matches are plotted since they are the rarer of the two classes. This peak inside the predictions
    # alongside the accuracy makes sure that model did not collapse. Colapsing is a big issue here, since the training
    # set is highly unbalanced in favour of missmatches.
    for i in range(len(x1)):
        if not pc[i]:
            print(y1[i],y2[i],'Same')

    # plot the loss curve.
    plt.plot(list(range(len(loss_list))),loss_list)
 #   plt.show()

    # CLASSIFICATION SEGMENT
    test_acc = []
    samples_per_shot = 1024# half_batch
    total_data_processed = 0.0
    correct = 0.0
    for data, labels in m.get_test_batch(batch_size):
        data = data.reshape((data.shape[0],28,28,1))
        print('[INFO] processing',total_data_processed,'of',m.get_sizes()[2])

        #calssify a single sample
        for i in range(len(data)):
            x1, y1 = m.get_samples(samples_per_shot)
            x1 = x1.reshape((x1.shape[0],28,28,1))
            x2 = np.asarray([list(data[i])] * samples_per_shot)
            #print(x2.shape)
            y2 = np.asarray([labels[i]] * samples_per_shot)
            y = np.zeros((samples_per_shot,1))
            y[y1 == y2] = 0.0
            y[y1 != y2] = 1.0
            pc = sess.run([predicted_class],feed_dict={X1:x1,X2:x2,Y:y})
            prediction = y1[np.argmin(pc)]
            #print('y:',prediction, 'label:',labels[i])
            if prediction == labels[i]:
                correct += 1.0
            total_data_processed += 1.0
        # keep track of loss and accuracy
    accuracy = correct / total_data_processed
    print('Classification Accuracy:', accuracy)
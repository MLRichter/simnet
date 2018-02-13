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
    sigmoidal_out = tf.nn.sigmoid(logits)
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
    v_loss_list = []
    v_acc_list = []
    i = 0
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

            if i%100 == 0:
                v_loss = []
                v_acc = []
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
                    v_loss.append(l)
                    v_acc.append(a)
                v_loss_list += [np.mean(v_loss)]*100
                v_acc_list += [np.mean(v_acc)]*100
            i += 1

    # print the mean accuracy over the test-set
    print('Acc:', np.mean(v_acc_list))

    # plot the labels of all pairs in the last batch that assumed to have the same label
    # only matches are plotted since they are the rarer of the two classes. This peak inside the predictions
    # alongside the accuracy makes sure that model did not collapse. Colapsing is a big issue here, since the training
    # set is highly unbalanced in favour of missmatches.
    for i in range(len(x1)):
        if not pc[i]:
            print(y1[i],y2[i],'Same')

    # plot the loss curve.
    plt.plot(list(range(len(loss_list))),loss_list, label='Train Loss')
    plt.plot(list(range(len(loss_list))),v_loss_list, label='Val Loss')
    plt.show()
    plt.plot(list(range(len(loss_list))),v_acc_list, label='Validation Accuracy')
    plt.show()

    def get_mean_prediction(predictions, y):
        score_map = []
        for i in range(10):
            c_pred = predictions[y == i]
            m = np.mean(c_pred)
            score_map.append(m)
        return np.argmin(score_map)

    # CLASSIFICATION SEGMENT
    test_acc = []
    samples_per_shot = 100
    total_data_processed = 0.0
    correct = 0.0
    correct_avg = 0.0
    for data, labels in m.get_test_batch(batch_size):
        data = data.reshape((data.shape[0],28,28,1))
        print('[INFO] processing',total_data_processed,'of',m.get_sizes()[2])

        #calssify a single sample
        for i in range(len(data)):
            x1, y1 = m.get_classification_samples(samples_per_shot // 10)
            x2 = np.asarray([list(data[i])] * samples_per_shot)

            pc = sess.run([sigmoidal_out],feed_dict={X1:x1,X2:x2})
            prediction = y1[np.argmin(pc)]
            prediction_avg = get_mean_prediction(np.squeeze(pc),y1)

            if prediction == labels[i]:
                correct += 1.0
            if prediction_avg == labels[i]:
                correct_avg += 1.0
            total_data_processed += 1.0
        # keep track of loss and accuracy
    accuracy = correct / total_data_processed
    avg_acc = correct_avg / total_data_processed
    print('Classification Accuracy:  ', accuracy)
    print('Averaging Sample Accuracy:', avg_acc)
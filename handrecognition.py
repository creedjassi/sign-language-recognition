import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
#import timeit
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.matmul(W,X)+b    
    sess = tf.compat.v1.Session()
    #tf.Session()
    result = sess.run(Y)
    sess.close()
    return result
print( "result = \n" + str(linear_function()))

def sigmoid(z):
    x = tf.compat.v1.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    sess =tf.compat.v1.Session()
    result = sess.run(sigmoid,feed_dict={x:z})
    return result
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
def cost(logits, labels):    
    z =  tf.compat.v1.placeholder(tf.float32,name='z')
    y =  tf.compat.v1.placeholder(tf.float32,name='y')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    sess = tf.compat.v1.Session()
    cost = sess.run(cost,feed_dict={z:logits,y:labels})
    sess.close()
    return cost
logits = np.array([0.2,0.4,0.7,0.9])   #for testing
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))
def one_hot_matrix(labels, C):
    C = tf.constant(C,name='C' )
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.compat.v1.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
labels = np.array([1,2,3,0,2,1])       #for testing
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = \n" + str(one_hot))
def ones(shape):
    ones = tf.ones(shape)
    sess = tf.compat.v1.Session()
    ones = sess.run(ones)
    sess.close()
    return ones
print ("ones = " + str(ones([3])))
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
print(Y_train_orig)


# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
def create_placeholders(n_x, n_y):
    X = tf.compat.v1.placeholder(tf.float32,[n_x,None],name='X')
    Y = tf.compat.v1.placeholder(tf.float32,[n_y,None],name='Y')
    return X, Y
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
def initialize_parameters():
    tf.set_random_seed(1)                  
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']              
    Z1 = tf.matmul(W1,X)+b1                                              
    A1 = tf.nn.relu(Z1)                                              
    Z2 = tf.matmul(W2,A1)+b2                                              
    A2 = tf.nn.relu(Z2)                                              
    Z3 = tf.matmul(W3,A2)+b3                                              
    return Z3
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits , labels =labels ))
    return cost
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    epoch_plot=[]      
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                             
    seed = 3                                          
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []                                        
    X, Y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer =tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            epoch_plot.append(epoch)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # ploting the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        plt.plot(np.squeeze(costs),epoch_plot)
        plt.show()
        
        #saving the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        #saver = tf.train.Saver([parameters])
        #saver.save(sess, 'my_test_model',global_step=1000)

        # Calculating the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculating accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        print(parameters['b1'])
        return parameters 
#start = timeit.timeit()
print("hello");
parameters = model(X_train, Y_train, X_test, Y_test)
history = model.fit(X_train, y_train, epochs=34, batch_size=1, validation_data=(X_val, y_val))
'''end = timeit.timeit()
print(end - start)'''

import scipy
from PIL import Image
from scipy import ndimage
import cv2
my_image = "45.jpg"
fname = "D:/datasetcollection/images/" + my_image
image = np.array(plt.imread(fname))
image = image/255.
my_image = cv2.resize(image,(64,64))
my_image = np.reshape(my_image,(1,64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))



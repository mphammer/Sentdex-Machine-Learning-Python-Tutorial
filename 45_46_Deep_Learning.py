import tensorflow as tf 
from tensorflow.keras.datasets import mnist

# turn off eager execution since instructor is using an older version of terraform
tf.compat.v1.disable_eager_execution()

'''
https://en.wikipedia.org/wiki/MNIST_database
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits 
that is commonly used for training various image processing systems. The database is also widely used for training and testing in
 the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets.
'''
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

n_nodes_hidden_layer_1 = 500
n_nodes_hidden_layer_2 = 500
n_nodes_hidden_layer_3 = 500

n_classes = 10 
batch_size = 100

x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', [None, 0])

def neural_network_model(data):
    hidden_1_layer = {
        "weights": tf.Variable(tf.compat.v1.random_normal([784, n_nodes_hidden_layer_1])),
        "biases": tf.Variable(tf.compat.v1.random_normal(n_nodes_hidden_layer_1)),
    }
    hidden_2_layer = {
        "weights": tf.Variable(tf.compat.v1.random_normal([n_nodes_hidden_layer_1, n_nodes_hidden_layer_2])),
        "biases": tf.Variable(tf.compat.v1.random_normal(n_nodes_hidden_layer_2)),
    }
    hidden_3_layer = {
        "weights": tf.Variable(tf.compat.v1.random_normal([n_nodes_hidden_layer_2, n_nodes_hidden_layer_3])),
        "biases": tf.Variable(tf.compat.v1.random_normal(n_nodes_hidden_layer_3)),
    }
    output_layer = {
        "weights": tf.Variable(tf.compat.v1.random_normal([n_nodes_hidden_layer_3, n_classes])),
        "biases": tf.Variable(tf.compat.v1.random_normal(n_classes)),
    }

    # (input_data * weights) + biases

    layer_1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]) + hidden_1_layer["biases"]) # multiple inputs and wights, and add bias
    layer_1 = tf.nn.relu(layer_1) # activation function

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer["weights"]) + hidden_2_layer["biases"]) # multiple inputs and wights, and add bias
    layer_2 = tf.nn.relu(layer_2) # activation function

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer["weights"]) + hidden_3_layer["biases"]) # multiple inputs and wights, and add bias
    layer_3 = tf.nn.relu(layer_3) # activation function

    output = tf.matmul(layer_3, output_layer["weights"]) + output_layer["biases"] # multiple inputs and wights, and add bias

    return output 

def train_neural_networ(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10 # cycles of forward propagation and back propagation 
    with tf.compat.v1.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            pass # TODO

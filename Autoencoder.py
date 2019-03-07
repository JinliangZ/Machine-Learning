import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# data preprocessing
data = pd.read_csv("D:/creditcard.csv")
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)


# Autoencoder
class Autoencoder(object):
    def __init__(self,n_hidden_1, n_hidden_2, n_input, learning_rate ):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_input = n_input
        self.learniing_rate = learning_rate
        self.weights, self.biases = self._initialize_weights()

        self.x = tf.placeholder("float", [None, self.n_input])  #define input
        self.encoder_op = self.encoder(self.x)  #features through encode
        self.decoder_op = self.decoder(self.encoder_op) #result after decode

        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op,2)) #cost function
        self.optimizer = tf.train.RMSPropOptimizer(self.learniing_rate).minimize(self.cost)
        # start session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
        }

        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
        }
        return weights, biases

    def encoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    def decoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})  #transform self.x to X

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})


#train test split
good_data = data[data['Class'] == 0]
bad_data = data[data['Class'] == 1]
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

X_train = X_train[X_train['Class']==0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values


X_good = good_data.ix[:, good_data.columns != 'Class']
y_good = good_data.ix[:, good_data.columns == 'Class']

X_bad = bad_data.ix[:, bad_data.columns != 'Class']
y_bad = bad_data.ix[:, bad_data.columns == 'Class']

# build model
model = Autoencoder(n_hidden_1 = 16,  n_hidden_2 = 4,n_input=X_train.shape[1], learning_rate = 0.01) # shape gets the #of features

#parameters
training_epochs = 5000 # read total data ? times
batch_size = 300  #300 data per batch
display_step = 50
record_step = 10

#train model
total_batch = int(X_train.shape[0] / batch_size) # sepearate data into several batches

cost_summary = []
total_cost2 = []
total_cost2.append(0)



for epoch in range(training_epochs):
    cost = None
    for i in range(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train[batch_start:batch_end, :]
        cost = model.partial_fit(batch)

    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)
        total_cost2.append(total_cost)
        #print(total_cost)
        #print(total_cost2)
        if abs(total_cost -total_cost2[-2]) < 0.001:
            print("converge", "cost=" , total_cost2[-1])
            break

        if epoch % record_step == 0 :
            cost_summary.append({'epoch': epoch + 1, 'cost': total_cost})


        if epoch % display_step == 0:
            print("Epoch:{}, cost={:.9f}".format(epoch + 1, total_cost))

# text model and get results
encode_decode = None
total_batch = int(X_test.shape[0]/batch_size) + 1
for i in range(total_batch):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = X_test[batch_start:batch_end, :]
    batch_res = model.reconstruct(batch)
    if encode_decode is None:
        encode_decode = batch_res
    else:
        encode_decode = np.vstack((encode_decode, batch_res)) #  add the array together in the vertical direction




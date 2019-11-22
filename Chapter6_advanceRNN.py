import numpy as np
import tensorflow as tf
#performing 


#define some constants
batch_size = 128;embedding_dimension = 64;num_classes = 2
hidden_layer_size = 32;times_steps = 6;element_size = 1

#create example sentences
digit_to_word_map = {1:"One",2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six",7:"Seven",8:"Eight",9:"Nine"}

#in this step, we pad sentences with zeros to make all senteces equally sized, process called zero padding
#this is accomplished in rozs 29 - 34
digit_to_word_map[0]="PAD"

even_sentences = []
odd_sentences = []


seqlens = []
for i in range(10000):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    #make two random chjoices between the integers of one and ten
    rand_odd_ints = np.random.choice(range(1,10,2), rand_seq_len)
    rand_even_ints = np.random.choice(range(1,10,2), rand_seq_len)

    #padding
    if rand_seq_len<6:
        rand_odd_ints = np.append(rand_odd_ints,[0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints,[0]*(6-rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

data = even_sentences+odd_sentences

# *= operator ex: x *= 3 is same as x = x * 3
seqlens*=2

even_sentences[0:6]
odd_sentences[0:6]

seqlens[0:6]


#splitting the sentence into indifivial words and adding each to the index map if not already present
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

#create inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
    label=labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]

labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

def get_sentence_batch(batch_size,data_x,data_y,data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens

_inputs = tf.placeholder(tf.int32, shape=[batch_size,times_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])


#creating word embedding
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                                                embedding_dimension],-1.0, 1.0),name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

#create the long short term memory celland feed it to a dynamic rnn, give dynamic rnn same sequence lengtyh using 
#_seqlens defined earlier
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length = _seqlens, 
                                        dtype=tf.float32)

    #use dictionar to assign linear layer as key to hidden layer size
    weights = {
        'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,
                                                         num_classes],
                                                        mean=0,stddev=.01))
    }

    Biases = {
        'linear_layer':tf.Variable(tf.truncated_normal([num_classes],
                                                       mean=0,stddev=.01))
    }

#take last valid output vector, this is available in states which was returned by dynamic_rnn
#then pass final output through a sofmax function 
final_output = tf.matmul(states[1],weights["linear_layer"]) + biases["linear_layer"]
softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output, labels=_labels
cross_entropy = tf.reduce_mean(softmax)

                                                  #begin training of classification model
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels,1),tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100

#Create sesssion
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                            train_x, train_y,
                                                            train_seqlens)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                        _seqlens: seqlen_batch})

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                _labels: y_batch,
                                                _seqlens: seqlen_batch})
            print("Accuracy at %d: %.5f" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

        output_example = sess.run([outputs], feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test})
        states_example = sess.run([states[1]], feed_dict={_inputs: x_test,
                                                      _labels: y_test,
                                                      _seqlens: seqlen_test})

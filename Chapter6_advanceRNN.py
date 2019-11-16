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

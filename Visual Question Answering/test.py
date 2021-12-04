
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[42]:


import os
import tensorflow as tf 
import numpy as np
import json
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, LSTM, Bidirectional, Lambda, Concatenate, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence 
from datetime import datetime


# In[6]:


# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']


# In[44]:


def create_csv(results, results_dir):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(str(key) + ',' + str(value) + '\n')


# In[8]:


def tokenize_data(texts, mxlen):
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    return seqs


# In[9]:


def tokenize_data(texts, mxlen):
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    return seqs


# In[7]:


def get_embeddings_index():
    embeddings_index = {}
    path = os.path.join(cwd,"glove.6B.50d.txt")
    f = open(path, 'r', errors='ignore',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


# In[8]:


def get_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[9]:


def get_embedding_layer(word_index, embedding_index, sequence_len):
    embedding_matrix = get_embedding_matrix(word_index, embedding_index)
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=sequence_len,
                     trainable=False)


# In[10]:


def bn_layer(num_filter, filter_size):
    def f(inputs):
        md = Conv2D(num_filter, (filter_size), padding='valid')(inputs)
        md = BatchNormalization()(md)
        return Activation('relu')(md)
    return f


def conv_net(inputs):
    model = bn_layer(24, 3)(inputs)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(24, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(24, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(24, 3)(model)
    model = MaxPooling2D((3, 3), 3)(model)
    model = bn_layer(24, 3)(model)
    return model


# In[12]:


#build g mlp network

def get_dense(n):
    r = []
    for k in range(n):
        r.append(Dense(mlp_unit, activation='relu'))
    return r


def get_MLP(n, denses):
    def g(x):
        d = x
        for k in range(n):
            d = denses[k](d)
        return d
    return g


# In[33]:


cwd=os.getcwd()
tokenizer=Tokenizer()
EMBEDDING_DIM=50
lstm_unit = 128
mlp_unit=256
cwd=os.getcwd()
dataset_dir=os.path.join(cwd,"dataset_vqa")
test_img_dir=os.path.join(dataset_dir,"test")
path=os.path.join(dataset_dir,"test_data.json")
f = open(path, 'r')
f=f.read()
f=json.loads(f)
data = []
mxlen=41
for jn in f['questions']:
    imgn = jn['image_filename']
    q = jn['question']
    qid=jn['question_id']
    data.append([q,imgn,qid])
    
ques_tok=tokenize_data([i[0] for i in data],mxlen)


# In[34]:


input1 = Input((160,240, 3))
input2 = Input((mxlen,))

cnn_features = conv_net(input1)

embedding_layer = get_embedding_layer(tokenizer.word_index,get_embeddings_index(), mxlen)
embedding = embedding_layer(input2)
bi_lstm = Bidirectional(LSTM(lstm_unit, implementation=2, return_sequences=False))
lstm_encode =  bi_lstm(embedding)

shapes = cnn_features.shape
w, h = shapes[1], shapes[2]
blocks = []
print(w,h)
for k1 in range(w):
    for k2 in range(h):
        def get_feature(t):
            return t[:, k1, k2, :]
        get_feature_block = Lambda(get_feature)
        blocks.append(get_feature_block(cnn_features))

pair_wise = []
concat = Concatenate()
for block1 in blocks:
    for block2 in blocks:
        pair_wise.append(concat([block1, block2, lstm_encode]))
        
g_MLP=get_MLP(4, get_dense(4))

gout = []
for p in pair_wise:
    gout.append(g_MLP(p))
added_out = Add()(gout)

#f mlp
f_mlp=Dense(mlp_unit,activation="relu")(added_out)
f_mlp=Dense(mlp_unit,activation="relu")(f_mlp)
f_mlp=Dropout(0.5)(f_mlp)
f_mlp_out=Dense(29,activation="relu")(f_mlp)

pred=Dense(13,activation="softmax")(f_mlp_out)

model = Model(inputs=[input1, input2], outputs=pred)
model.load_weights(os.path.join(cwd,"experiments_dir\\relation_net_Jan19_16-24-11\\ckpts\\mymodel_1.h5"))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[35]:


def load_image(image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img=cv2.resize(img,(240,160))
        img = img / 255
        return img


# In[45]:


results={}
for pair in data:
    img_path=os.path.join(test_img_dir,pair[1])
    img=load_image(img_path)
    np_image=np.expand_dims(img, axis=0)
    prediction=model.predict([np_image,np.expand_dims(ques_tok[pair[2]],axis=0)])
    results[pair[2]] = np.argmax(prediction)
    print(pair[2])
results_dir=os.path.join(cwd,"results")
create_csv(results,results_dir)


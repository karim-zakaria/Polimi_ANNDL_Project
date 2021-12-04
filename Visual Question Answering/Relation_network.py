
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


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


# In[3]:


EMBEDDING_DIM=50
lstm_unit = 128
mlp_unit=256
mxlen=0
cwd=os.getcwd()
tokenizer=Tokenizer()
ans_label={'0': 0,
           '1': 1,
           '10': 2,
           '2': 3,
           '3': 4,
           '4': 5,
           '5': 6,
           '6': 7,
           '7': 8,
           '8': 9,
           '9': 10,
           'no': 11,
           'yes': 12
           }


# In[4]:


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,
                 list_IDs,
                 list_ques,
                 list_img, 
                 labels,
                 image_path,
                 batch_size, 
                 dim, 
                 n_classes, 
                 mxlen,
                 n_channels=3,
                 to_fit=True, 
                 shuffle=True):
        
        self.list_IDs=list_IDs
        self.list_ques=list_ques
        self.list_img = list_img
        self.labels = labels
        self.image_path = image_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.n_channels=n_channels
        self.shuffle = shuffle
        self.mxlen=mxlen
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_ques) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X1,X2 = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return [X1,X2], y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X1= np.empty((self.batch_size, *self.dim, self.n_channels))
        X2=np.empty((self.batch_size,self.mxlen), dtype=object)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = self._load_image(os.path.join(self.image_path,self.list_img[ID]))
            X2[i,] = self.list_ques[ID]
            
        return X1,np.asarray(X2, dtype=np.float32)

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size,1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self.labels[ID]
        return y

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img=cv2.resize(img,(240,160))
        img = img / 255
        return img


# In[5]:


def load_data(path):
    f = open(path, 'r')
    f=f.read()
    f=json.loads(f)
    data = []
    qid=0
    mxlen=0
    for jn in f['questions']:
        imgn = jn['image_filename']
        la = ans_label[jn['answer']]
        s = jn['question']
        if len(s.split())>mxlen:
            mxlen=len(s.split())
        data.append([qid,s,imgn, la])
        qid+=1
    return data,mxlen


# In[6]:


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


dataset_dir=os.path.join(cwd,"dataset_vqa")
train_img_dir=os.path.join(dataset_dir,"train")
train_data,mxlen=load_data(os.path.join(dataset_dir,"train_data.json"))
img_size=(160,240)
ques=[i[1] for i in train_data]
tok_ques=tokenize_data(ques,mxlen)
bs=48

train_generator= DataGenerator(list_IDs=[i[0] for i in train_data],
                               list_ques=tok_ques,
                               list_img=[i[2] for i in train_data], 
                               labels=[i[3] for i in train_data],
                               image_path=train_img_dir,
                               batch_size=bs, 
                               dim=img_size, 
                               n_classes=13,
                               mxlen=mxlen)


# In[11]:


#build cnn

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


# In[13]:


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


# In[14]:


# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.load_weights(os.path.join(cwd,"experiments_dir\\relation_net_Jan17_07-19-54\\ckpts\\mymodel_11.h5"))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model_json = model.to_json()
with open("relation_net.json", "w") as json_file:
    json_file.write(model_json)


# In[15]:


import os
from datetime import datetime

cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'experiments_dir')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_name = 'relation_net'

exp_dir = os.path.join(exps_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    
callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(ckpt_dir,'mymodel_{epoch}.h5'),
              save_best_only=False,
              save_weights_only=True,
              verbose=1)

callbacks.append(ckpt_callback)

# ----------------

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
    
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)


# ---------------------------------

from tensorflow.keras.callbacks import ReduceLROnPlateau
lrr = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=2, min_lr=0.00001, verbose=1)
callbacks.append(lrr)

model.fit_generator(generator=train_generator,
          epochs=100,
          callbacks=callbacks)

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser


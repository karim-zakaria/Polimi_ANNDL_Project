
# coding: utf-8

# In[14]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import numpy as np

# Set the seed for random operations. 
# This let our experiments to be reproducible. 
SEED = 1234
tf.random.set_seed(SEED)  

# Get current working directory
cwd = os.getcwd()


# In[16]:


# ImageDataGenerator
# ------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input

apply_data_augmentation = True

# Create training ImageDataGenerator object
# We need two different generators for images and corresponding masks
if apply_data_augmentation:
    train_img_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            preprocessing_function=preprocess_input)
    train_mask_data_gen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=10,
                                             height_shift_range=10,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='constant',
                                             cval=0)
else:
    train_img_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_mask_data_gen = ImageDataGenerator()

# Create validation and test ImageDataGenerator objects
valid_img_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_mask_data_gen = ImageDataGenerator()


# In[67]:


# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join(cwd, 'Dataset1')

# Batch size
bs = 5

# img shape
img_h = 256
img_w = 256

num_classes=21

# Training
# Two different generators for images and masks
# ATTENTION: here the seed is important!! We have to give the same SEED to both the generator
# to apply the same transformations/shuffling to images and corresponding masks
training_dir = os.path.join(cwd, 'Dataset1\\training')
print(training_dir)
train_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED)  
train_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED)
train_gen = zip(train_img_gen, train_mask_gen)

# Validation
validation_dir = os.path.join(dataset_dir, 'validation')
valid_img_gen = valid_img_data_gen.flow_from_directory(os.path.join(validation_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs,
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=False,
                                                       interpolation='bilinear',
                                                       seed=SEED)
valid_mask_gen = valid_mask_data_gen.flow_from_directory(os.path.join(validation_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=False,
                                                         interpolation='bilinear',
                                                         seed=SEED)
valid_gen = zip(valid_img_gen, valid_mask_gen)


# In[18]:


#Create dataset objects

# Training
# --------
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 3]))

def prepare_target(x_, y_):
    y_ = tf.cast(tf.expand_dims(y_[..., 0], -1), tf.int32)
    return x_, tf.where(y_ > 0, y_ - 1, y_ + 1)

train_dataset = train_dataset.map(prepare_target)

# Repeat
train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 3]))
valid_dataset = valid_dataset.map(prepare_target)

# Repeat
valid_dataset = valid_dataset.repeat()


# In[5]:


#Import encoder model
densenet=tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
densenet.summary()


# In[55]:


#Create model
#Add densenet backbone

finetuning = False

if finetuning:
    freeze_until = 15 # layer from which we want to fine-tune
    
    for layer in densenet.layers[:freeze_until]:
        layer.trainable = False
else:
    densenet.trainable = False



conv6=tf.keras.layers.Conv2D(filters=1024,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(densenet.get_layer("relu").output)

#Upsampling block 1
conc1=tf.keras.layers.Concatenate(axis=-1)([conv6,densenet.get_layer("pool4_pool").output])

conv7=tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(conc1)
up1=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(conv7)

#Upsampling block 2
conc2=tf.keras.layers.Concatenate(-1)([up1,densenet.get_layer("pool3_pool").output])

conv8=tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(conc2)
up2=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(conv8)

#Upsampling block 3
conc3=tf.keras.layers.Concatenate(-1)([up2,densenet.get_layer("pool2_pool").output])

conv9=tf.keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(conc3)
up3=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(conv9)

#Upsampling block 4
conc4=tf.keras.layers.Concatenate(-1)([up3,densenet.get_layer("pool1").output])

conv10=tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(conc4)
up4=tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(conv10)

#Mix and predict

conv11=tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            activity_regularizer=tf.keras.regularizers.l1(0.01))(up4)
output=tf.keras.layers.Conv2D(filters=2,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='softmax')(conv11)

model=tf.keras.Model(densenet.input,output)
model_json=model.to_json()
with open("model_densenet_backbone.json", "w") as json_file:
    json_file.write(model_json)


# In[56]:


# Optimization params
# -------------------

# Loss
# Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[ ]:


import os
from datetime import datetime

cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'experiments_dir')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'model4'
temp_string=model_name+str(now)
exp_dir = os.path.join(exps_dir, temp_string)
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
              monitor='val_loss',
              verbose=1)

callbacks.append(ckpt_callback)

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

# Early Stopping
# --------------
early_stop = False
if early_stop:
    es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=10)
    callbacks.append(es_callback)


model.fit(x=train_dataset,
          epochs=100,  #### set repeat in training dataset
          steps_per_epoch=len(train_img_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_img_gen), 
          callbacks=callbacks)


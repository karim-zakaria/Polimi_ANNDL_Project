
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import os
import tensorflow as tf
import numpy as np

# Set the seed for random operations. 
# This let our experiments to be reproducible. 
SEED = 1234
tf.random.set_seed(SEED)  

# Get current working directory
cwd = os.getcwd()



# In[3]:


# ImageDataGenerator
# ------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = True

# Create training ImageDataGenerator object
#model3 no rotation
if apply_data_augmentation:
      train_data_gen= ImageDataGenerator(rotation_range=50,
                                        shear_range=0.15,
                                        horizontal_flip=True,
                                        fill_mode="nearest",
                                        width_shift_range=10,
                                        height_shift_range=10,
                                        zoom_range=0.3,
                                        rescale=1./255)
else:
    train_data_gen= ImageDataGenerator(rescale=1./255)
    
# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)


# In[4]:


# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join(cwd, 'new_dataset3')

# Batch size
bs = 10

# img shape
img_h = 256
img_w = 256

num_classes=20

class_list=['owl',
    'galaxy',
    'lightning',
    'wine-bottle',
    't-shirt',
    'waterfall',
    'sword',
    'school-bus',
    'calculator',
    'sheet-music',
    'airplanes',
    'lightbulb',
    'skyscraper',
    'mountain-bike',
    'fireworks',
    'computer-monitor',
    'bear',
    'grand-piano',
    'kangaroo',
    'laptop']

# Training
training_dir = os.path.join(dataset_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs, 
                                               classes=class_list,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED) 

# Validation
validation_dir = os.path.join(dataset_dir, 'validation')
valid_gen = valid_data_gen.flow_from_directory(validation_dir,
                                               batch_size=bs, 
                                               classes=class_list,
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=SEED)


# In[5]:


# Create Dataset objects
# ----------------------

# Training
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Repeat
valid_dataset = valid_dataset.repeat()


# In[ ]:


#load inception module
resnet_inception=tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))


# In[ ]:


#model definition

finetuning = True

if finetuning:
    freeze_until = 10 # layer from which we want to fine-tune
    
    for layer in resnet_inception.layers[:freeze_until]:
        layer.trainable = False
else:
    inception.trainable = False

#model3 only one hidden fc layer & dropout
model = tf.keras.Sequential()
model.add(resnet_inception)
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


# In[ ]:


# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
#model3 constant learning rate

lr=1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model_json=model.to_json()
with open("model5.json", "w") as json_file:
    json_file.write(model_json)

# In[ ]:


import os
from datetime import datetime

cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'experiments_dir')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'model5'
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
              save_best_only=True,
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
#early_stop = False
#if early_stop:
#    es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=10)
#    callbacks.append(es_callback)


model.fit(x=train_dataset,
          epochs=50,  #### set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_gen), 
          callbacks=callbacks)


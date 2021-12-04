
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
if apply_data_augmentation:
      train_data_gen= ImageDataGenerator(rotation_range=20,
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
dataset_dir = os.path.join(cwd, 'new_dataset')

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
inception=tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))


# In[ ]:


#model definition

finetuning = True

if finetuning:
    freeze_until = 10 # layer from which we want to fine-tune
    
    for layer in inception.layers[:freeze_until]:
        layer.trainable = False
else:
    inception.trainable = False

#model2 only one hidden fc layer
model = tf.keras.Sequential()
model.add(inception)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


# In[ ]:


# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
#model2 constant learning rate
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
with open("model2.json", "w") as json_file:
    json_file.write(model_json)


# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import os
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

cwd=os.getcwd()


# In[3]:


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy']


# In[4]:


json_file=open('model3.json','r')
loaded_model=json_file.read()
json_file.close()
model=tf.keras.models.model_from_json(loaded_model)
model.load_weights(os.path.join(cwd,"experiments_dir\\model3Nov24_03-34-57\\ckpts\\mymodel_37.h5"))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()


# In[5]:


def create_csv(results, results_dir):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


# In[6]:


test_set=os.path.join(cwd,"new_dataset2\\test")
results={}
for img_name in os.listdir(test_set):
    img_path=os.path.join(test_set,img_name)
    img=tf.keras.preprocessing.image.load_img(img_path,target_size=(256,256))
    np_image=np.expand_dims(img, axis=0)
    np_image=np_image/255.0
    prediction=model.predict_classes(np_image)
    print(img_path)
    print(prediction[0])
    results[img_name] = prediction[0]

results_dir=os.path.join(cwd,"results\\model3")
create_csv(results,results_dir)


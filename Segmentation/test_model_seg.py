
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
from tensorflow.keras.applications.resnet import preprocess_input
from matplotlib import pyplot as plt

cwd=os.getcwd()


# In[3]:


loss = tf.keras.losses.BinaryCrossentropy() 
# learning rate
lr = 1e-6
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ["accuracy"]


# In[4]:


json_file=open('model_Unet.json','r')
loaded_model=json_file.read()
json_file.close()
model=tf.keras.models.model_from_json(loaded_model)
model.load_weights(os.path.join(cwd,"experiments_dir\\model_resnet_UnetDec18_00-31-12\\ckpts\\mymodel_1.h5"))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[5]:


def rle_encode(img):
      # Flatten column-wise
      pixels = img.T.flatten()
      pixels = np.concatenate([[0], pixels, [0]])
      runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
      runs[1::2] -= runs[::2]
      return ' '.join(str(x) for x in runs)


# In[6]:


import os
from datetime import datetime

def create_csv(results, results_dir):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir,csv_fname), 'w') as f:

      f.write('ImageId,EncodedPixels,Width,Height\n')

      for key, value in results.items():
          f.write(key + ',' + str(value) + ',' + '256' + ',' + '256' + '\n')


# In[ ]:


test_img_dir=os.path.join(cwd,"Dataset1\\test\\images\\img")
results={}
results_dir=os.path.join(cwd,"results\\model_resnet")

for img_name in os.listdir(test_img_dir):
    
      img = Image.open(os.path.join(test_img_dir, img_name))   
      img = img.resize((256,256))
      img_arr = np.expand_dims(np.array(img),0)
      img_arr=preprocess_input(img_arr)
      prediction_mask = model.predict(x = img_arr)
      prediction_mask = (prediction_mask > 0.5)
      mask = prediction_mask[0]
      mask_pred = rle_encode(prediction_mask)
      name = os.path.splitext(img_name)[0]
      predicted_mask=prediction_mask[0].squeeze()
      plt.imsave(os.path.join(cwd,"pred_test",img_name),img)
      plt.imsave(os.path.join(cwd,"pred_test",name+".png"),predicted_mask)
      print(name)
      results[name] = mask_pred  

create_csv(results,results_dir)


# In[ ]:


#    img_arr = np.expand_dims(np.array(img), 0)
#    img_arr=preprocess_input(img_arr)
#    predicted_mask= model.predict(x=img_arr/.255)
#    image_id=os.path.splitext(img_name)[0]
#    print(image_id)
#    st= predicted_mask < 0.8
#    predicted_mask[st] = 0
#    predicted_mask=np.round(predicted_mask[0]).squeeze()
#    plt.imsave(os.path.join(cwd,"pred_test",image_id+".png"),predicted_mask)
#    encoded_mask=rle_encode(predicted_mask)
#    results[image_id]=encoded_mask


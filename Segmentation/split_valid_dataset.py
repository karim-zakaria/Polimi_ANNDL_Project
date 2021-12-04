
# coding: utf-8

# In[63]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[64]:


import os
import shutil
cwd=os.getcwd()


#create validation directories
c=0
trainimg=os.path.join(cwd,'Dataset1\\training\\images')
trainmsk=os.path.join(cwd,'Dataset1\\training\\masks')
validimg=os.path.join(cwd,'Dataset1\\validation\\images')
validmsk=os.path.join(cwd,'Dataset1\\validation\\masks')

for file in os.listdir(os.path.join(trainimg)):
        if c%50==0:
            os.replace(os.path.join(trainimg ,file),os.path.join(validimg ,file))
            os.replace(os.path.join(trainmsk ,file),os.path.join(validmsk,file))
        c=c+1



# coding: utf-8

# In[63]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[64]:


import os
import shutil
cwd=os.getcwd()


# In[65]:


#create new dataset directories
oldDS=os.path.join(cwd,'Classification_Dataset')
newDS=os.path.join(cwd,'new_dataset3')
shutil.copytree(oldDS, newDS)


# In[66]:


#create validation dir
newValid=os.path.join(cwd,'new_dataset3\\validation')
os.mkdir(newValid)


# In[68]:


#create validation directories
c=0
newTrain=os.path.join(newDS,'training')
for name in os.listdir(newTrain):
    temppath=os.path.join(newValid,name)
    os.mkdir(temppath)
    for file in os.listdir(os.path.join(newTrain,name)):
        if c%20==0:
            os.replace(os.path.join(newTrain,name,file),os.path.join(temppath,file))
        c=c+1


import os
import json

dataset_path=os.path.join(os.getcwd(),'new_dataset2')
def path_to_dict(path):
    d = {'name': os.path.basename(path)}
    if os.path.isdir(path):
        d['type'] = "directory"
        d['children'] = [path_to_dict(os.path.join(path,x)) for x in os.listdir\
(path)]
    else:
        d['type'] = "file"
    return d

with open("dataset_split.json", "w") as json_file:
    json_file.write(json.dumps(path_to_dict(dataset_path)))

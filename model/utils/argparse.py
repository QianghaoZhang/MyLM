import json
def get_labels(path):
    with open(path,"r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels =['O'] + labels
    return labels

label_path = "/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/trained_data/labels.txt"
labels = get_labels(label_path)
num_labels = len(labels)
class AttrDict(dict):
    def __init__(self,*args,**kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__ = self
with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/dataset_train/config.json",'r') as f:
    args = json.load(f)

args['num_labels']=num_labels
args['has_visual_segment_embedding']=True
args["convert_sync_batchnorm"] =True
config = AttrDict(args)

if __name__=="__main__":
    with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/dataset_train/config.json",'r') as f:
        args = json.load(f)
        config = AttrDict(args)
        print(config)


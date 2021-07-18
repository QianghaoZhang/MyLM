import json


class AttrDict(dict):
    def __init__(self,*args,**kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__ = self
with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/dataset_train/config.json",'r') as f:
    args = json.load(f)
    config = AttrDict(args)

if __name__=="__main__":
    with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/dataset_train/config.json",'r') as f:
        args = json.load(f)
        config = AttrDict(args)
        print(config.vocab_size)


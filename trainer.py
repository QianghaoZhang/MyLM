import torch.nn as nn
from model.embedding.layoutlmv2 import LayoutLMv2Embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.utils.argparse import  config
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
torch.manual_seed(1)
class LanguageModeler(nn.Module):
    def __init__(self,config):
        super(LanguageModeler,self).__init__()
        self.config = config
        self.embeddings = LayoutLMv2Embeddings(config)
        self.linear1 = nn.Linear(config.coordinate_size*6,128)
        self.linear2 = nn.Linear(128,config.max_position_embeddings)
    def forward(self,bbox):
        position_embeds = self.embeddings._cal_spatial_position_embeddings(bbox)
        print(position_embeds.shape)
        out = F.relu(self.linear1(position_embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


if __name__=="__main__":
    with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/trained_data/bbox.txt",'r',encoding="utf-8") as fr:
        bbox_data = fr.read()
    with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/trained_data/labels.txt",'r',encoding="utf-8") as fr:
        labels_data = fr.read()
    inps = torch.tensor(eval(bbox_data))
    tgts = torch.tensor(eval(labels_data))

    dataset = TensorDataset(inps,tgts)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)

    loss_function = nn.NLLLoss()
    model = LanguageModeler(config)
    optimizer = optim.SGD(model.parameters(),lr = 0.01)
    losses = []
    for epoch in range(2):
        total_loss = 0

        for batch in tqdm(loader):
            model.zero_grad()
            log_probs = model(batch.inp)
            loss = loss_function(log_probs,batch.tgt)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
        losses.append(total_loss)
    print(losses)
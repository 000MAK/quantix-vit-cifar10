import argparse, torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=1); p.add_argument('--limit',type=int,default=5000); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
 root=Path(__file__).parent; (root/'data').mkdir(exist_ok=True); (root/'outputs').mkdir(exist_ok=True)
 tf=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
 ds=datasets.CIFAR10(root/'data',train=True,download=True,transform=tf); n=min(a.limit,len(ds)); ds=Subset(ds,range(n)); trn,val=random_split(ds,[int(n*.8),n-int(n*.8)],generator=torch.Generator().manual_seed(42))
 test=datasets.CIFAR10(root/'data',train=False,download=True,transform=tf); test=Subset(test,range(min(2000,len(test))))
 device='cuda' if torch.cuda.is_available() else 'cpu'
 class ViT(nn.Module):
  def __init__(self):
   super().__init__(); self.patch=nn.Conv2d(3,96,8,8); self.cls=nn.Parameter(torch.zeros(1,1,96)); self.pos=nn.Parameter(torch.randn(1,17,96)*.02); self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(96,4,192,batch_first=True),2); self.head=nn.Linear(96,10)
  def forward(self,x):
   z=self.patch(x).flatten(2).transpose(1,2); z=torch.cat((self.cls.expand(len(x),-1,-1),z),1)+self.pos; return self.head(self.enc(z)[:,0])
 m=ViT().to(device); opt=torch.optim.AdamW(m.parameters(),lr=5e-4); lossfn=nn.CrossEntropyLoss()
 def epoch(dl,train):
  m.train(train); total=ok=cnt=0
  for x,y in dl:
   x,y=x.to(device),y.to(device); opt.zero_grad(); out=m(x); loss=lossfn(out,y)
   if train: loss.backward(); opt.step()
   total+=loss.item()*len(y); ok+=(out.argmax(1)==y).sum().item(); cnt+=len(y)
  return {'loss':round(total/cnt,4),'accuracy':round(ok/cnt,4)}
 result={};
 for i in range(a.epochs): result[f'epoch_{i+1}']={'train':epoch(DataLoader(trn,batch_size=a.batch_size,shuffle=True),'train'),'validation':epoch(DataLoader(val,batch_size=a.batch_size),'eval')}
 result['test']=epoch(DataLoader(test,batch_size=a.batch_size),'eval'); torch.save(m.state_dict(),root/'outputs/model.pt'); import json; (root/'outputs/metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

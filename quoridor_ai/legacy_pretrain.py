import argparse,torch,random,json
from pathlib import Path
import numpy as np
from .model import PolicyValueNet

def load_legacy(folder,limit=15000):
 data=[]
 for p in sorted(Path(folder).rglob('*_legacy.pt')):
  d=torch.load(p,map_location='cpu',weights_only=False)
  for x,pi,z in d.get('replay',[]):
   y=np.zeros((11,9,9),np.float32);y[:7]=np.asarray(x,dtype=np.float32);data.append((y,np.asarray(pi,dtype=np.float32),float(z)))
   if len(data)>=limit:return data
 return data

def run(folder,output,channels=96,blocks=8,epochs=5,batch=512):
 data=load_legacy(folder);assert data,'No legacy replay found';device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=PolicyValueNet(channels,blocks).to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
 for ep in range(epochs):
  random.shuffle(data);pl=vl=0;n=0
  for i in range(0,len(data),batch):
   b=data[i:i+batch];x=torch.from_numpy(np.stack([q[0] for q in b])).to(device);pi=torch.from_numpy(np.stack([q[1] for q in b])).to(device);z=torch.tensor([q[2] for q in b],device=device)
   with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,v=net(x);lp=-(pi*torch.log_softmax(logits,1)).sum(1).mean();lv=torch.nn.functional.mse_loss(v,z);loss=lp+lv
   opt.zero_grad();loss.backward();opt.step();pl+=lp.item();vl+=lv.item();n+=1
  print(f'legacy epoch={ep} policy={pl/n:.4f} value={vl/n:.4f}',flush=True)
 out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':net.state_dict(),'legacy_samples':len(data),'channels':channels,'blocks':blocks},out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',default='legacy');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch',type=int,default=512);a=p.parse_args();run(a.legacy,a.output,epochs=a.epochs,batch=a.batch)
if __name__=='__main__':main()

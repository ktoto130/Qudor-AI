import math,numpy as np,torch
from .core.engine import legal_actions,apply_unchecked,ACTION_SIZE
from .core.encoding import encode_batch
class Node:
 def __init__(self,s,p=1.):self.s=s;self.p=p;self.n=0;self.w=0.;self.children={}
 @property
 def q(self):return self.w/self.n if self.n else 0.
def _infer(net,nodes,device):
 x=torch.from_numpy(encode_batch([n.s for n in nodes])).to(device)
 with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,values=net(x)
 for i,n in enumerate(nodes):
  acts=legal_actions(n.s);z=logits[i,acts].float().cpu().numpy();z-=z.max();p=np.exp(z);p/=p.sum();n.children={a:Node(apply_unchecked(n.s,a),float(v)) for a,v in zip(acts,p)}
 return values.float().cpu().numpy()
def batched_search(net,states,device,sims=64,c_puct=1.5):
 roots=[Node(s) for s in states];_infer(net,roots,device)
 for _ in range(sims):
  leaves=[];paths=[]
  for root in roots:
   n=root;path=[]
   while n.children:
    _,n=max(n.children.items(),key=lambda kv:-kv[1].q+c_puct*kv[1].p*math.sqrt(n.n+1)/(1+kv[1].n));path.append(n)
   leaves.append(n);paths.append(path)
  live=[n for n in leaves if n.s.winner is None]
  vals_iter=iter(_infer(net,live,device) if live else [])
  for leaf,path in zip(leaves,paths):
   v=-1. if leaf.s.winner is not None else float(next(vals_iter));leaf.n+=1;leaf.w+=v
   for p in reversed(path):v=-v;p.n+=1;p.w+=v
 out=[]
 for root in roots:
  pi=np.zeros(ACTION_SIZE,np.float32)
  for a,n in root.children.items():pi[a]=n.n
  if pi.sum():pi/=pi.sum()
  out.append(pi)
 return np.stack(out)

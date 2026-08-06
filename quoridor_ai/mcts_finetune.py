import argparse,json,random,torch
from pathlib import Path
import numpy as np
from .model import PolicyValueNet
from .core.engine import State,apply_unchecked
from .core.encoding import encode
from .batched_mcts import batched_search

def run(config,checkpoint,output,games=16,sims=64,steps=100):
 c=json.load(open(config));d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=PolicyValueNet(c['channels'],c['blocks']).to(d);ck=torch.load(checkpoint,map_location=d,weights_only=False);net.load_state_dict(ck['model']);states=[State() for _ in range(games)];trajectories=[[] for _ in states];done=[False]*games
 while not all(done):
  ids=[i for i,x in enumerate(states) if not done[i]];pis=batched_search(net,[states[i] for i in ids],d,sims=sims)
  for j,i in enumerate(ids):
   pi=pis[j];a=int(np.random.choice(len(pi),p=pi));trajectories[i].append((encode(states[i]),pi,states[i].player));states[i]=apply_unchecked(states[i],a);done[i]=states[i].winner is not None or states[i].ply>=c['max_plies']
  print(f'mcts active={len(ids)} total_positions={sum(map(len,trajectories))}',flush=True)
 data=[]
 for s,tr in zip(states,trajectories):data += [(x,pi,0 if s.winner is None else (1 if s.winner==pl else -1)) for x,pi,pl in tr]
 opt=torch.optim.AdamW(net.parameters(),lr=c['lr']*.25)
 for _ in range(steps):
  b=random.sample(data,min(c['batch'],len(data)));x=torch.tensor(np.stack([q[0] for q in b]),device=d);pi=torch.tensor(np.stack([q[1] for q in b]),device=d);z=torch.tensor([q[2] for q in b],dtype=torch.float32,device=d);logits,v=net(x);loss=-(pi*torch.log_softmax(logits,1)).sum(1).mean()+torch.nn.functional.mse_loss(v,z);opt.zero_grad();loss.backward();opt.step()
 Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save({'model':net.state_dict(),'config':c,'stage':'mcts','positions':len(data)},output);print(f'mcts finetune saved {output}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--games',type=int,default=16);p.add_argument('--sims',type=int,default=64);a=p.parse_args();run(a.config,a.checkpoint,a.output,a.games,a.sims)
if __name__=='__main__':main()

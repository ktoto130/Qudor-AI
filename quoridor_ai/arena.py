import argparse,torch,random,json,math
from pathlib import Path
import numpy as np
from .model import PolicyValueNet
from .core.engine import State,legal_actions,apply_unchecked
from .core.encoding import encode_batch

def load(path,device):
 d=torch.load(path,map_location=device,weights_only=False);c=d.get('config',{});m=PolicyValueNet(c.get('channels',d.get('channels',96)),c.get('blocks',d.get('blocks',8))).to(device);m.load_state_dict(d['model']);m.eval();return m
def choose(m,s,d):
 a=legal_actions(s);x=torch.from_numpy(encode_batch([s])).to(d)
 with torch.inference_mode():z,_=m(x)
 return a[int(z[0,a].argmax())]
def play(a,b,d,swap=False):
 s=State()
 while s.winner is None and s.ply<220:
  m=(a if s.player==0 else b) if not swap else (b if s.player==0 else a);s=apply_unchecked(s,choose(m,s,d))
 if s.winner is None:return .5
 winner_a=(s.winner==0) != swap;return 1. if winner_a else 0.
def run(candidate,best,games,out):
 d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a=load(candidate,d);b=load(best,d);scores=[play(a,b,d,i%2==1) for i in range(games)];wr=sum(scores)/games;elo=400*math.log10(max(1e-4,wr)/max(1e-4,1-wr));r={'games':games,'wins':sum(x==1 for x in scores),'draws':sum(x==.5 for x in scores),'win_rate':wr,'elo_delta':elo};Path(out).write_text(json.dumps(r,indent=2));print(r)
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--best',required=True);p.add_argument('--games',type=int,default=40);p.add_argument('--output',default='arena.json');a=p.parse_args();run(a.candidate,a.best,a.games,a.output)
if __name__=='__main__':main()

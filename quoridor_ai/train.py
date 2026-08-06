import argparse,csv,json,random,time,os
from pathlib import Path
from collections import deque
import numpy as np,torch
from .model import PolicyValueNet
from .selfplay import batched_selfplay

def run(config,output,resume=True,init=None):
 c=json.load(open(config));seed=c.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(output);out.mkdir(parents=True,exist_ok=True)
 net=PolicyValueNet(c['channels'],c['blocks']).to(device,memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');replay=deque(maxlen=c['replay']);start=0;ck=out/'latest.pt'
 if resume and ck.exists():
  d=torch.load(ck,map_location=device,weights_only=False);net.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);scaler.load_state_dict(d.get('scaler',{}));replay.extend(d.get('replay',[]));start=d['iteration']+1
 elif init and Path(init).exists():
  d=torch.load(init,map_location=device,weights_only=False);net.load_state_dict(d['model']);print(f'initialized from {init}',flush=True)
 metrics=out/'metrics.csv'
 if not metrics.exists():metrics.write_text('iteration,stage,games,positions,replay,games_per_sec,positions_per_sec,avg_length,policy_loss,value_loss,total_loss,seconds,device,vram_mb\n')
 for it in range(start,c['iterations']):
  t=time.time();data,sp=batched_selfplay(net,device,c['games'],c['max_plies'],c['inference_batch']);replay.extend(data)
  net.train();pl=vl=0.;steps=c['steps']
  for _ in range(steps):
   batch=random.sample(list(replay),min(c['batch'],len(replay)));x=torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device,memory_format=torch.channels_last);pi=torch.from_numpy(np.stack([b[1] for b in batch])).float().to(device);z=torch.tensor([b[2] for b in batch],dtype=torch.float32,device=device)
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,val=net(x);lp=-(pi*torch.log_softmax(logits,1)).sum(1).mean();lv=torch.nn.functional.mse_loss(val,z);loss=lp+lv
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),5);scaler.step(opt);scaler.update();pl+=lp.item();vl+=lv.item()
  sec=time.time()-t;vram=torch.cuda.max_memory_allocated()/1048576 if device.type=='cuda' else 0;row=[it,'pretrain',sp['games'],sp['positions'],len(replay),sp['games']/sp['seconds'],sp['positions']/sp['seconds'],sp['avg_length'],pl/steps,vl/steps,(pl+vl)/steps,sec,str(device),vram]
  with metrics.open('a',newline='') as f:csv.writer(f).writerow(row)
  payload={'iteration':it,'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'replay':list(replay)[-c['checkpoint_replay']:],'config':c};tmp=out/'latest.tmp';torch.save(payload,tmp);os.replace(tmp,ck);(out/'status.json').write_text(json.dumps(dict(zip(metrics.read_text().splitlines()[0].split(','),row)),indent=2));print(f"iteration={it} games/s={row[5]:.2f} pos/s={row[6]:.1f} loss={row[10]:.4f} device={device} vram={vram:.0f}MB",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--init');p.add_argument('--no-resume',action='store_true');a=p.parse_args();run(a.config,a.output,not a.no_resume,a.init)
if __name__=='__main__':main()

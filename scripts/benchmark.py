import time,torch
from quoridor_ai.core.engine import State,legal_actions,apply_unchecked
from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.model import PolicyValueNet
s=State();n=200;t=time.time()
for _ in range(n):a=legal_actions(s);s=apply_unchecked(s,a[0]);s=State() if s.winner is not None else s
print(f'engine: {n/(time.time()-t):.1f} positions/s')
d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=PolicyValueNet(64,4).to(d).eval();x=torch.from_numpy(encode_batch([State()]*512)).to(d)
for _ in range(3):m(x)
if d.type=='cuda':torch.cuda.synchronize()
t=time.time()
for _ in range(20):m(x)
if d.type=='cuda':torch.cuda.synchronize()
print(f'inference: {512*20/(time.time()-t):.0f} states/s device={d} vram={torch.cuda.max_memory_allocated()/1048576 if d.type=="cuda" else 0:.0f}MB')

import numpy as np
from .engine import State,rc
PLANES=11
def encode(s:State):
 x=np.zeros((PLANES,9,9),np.float32);r,c=rc(s.p0);x[0,r,c]=1;r,c=rc(s.p1);x[1,r,c]=1
 x[2].fill(s.player);x[3].fill(s.walls0/10);x[4].fill(s.walls1/10)
 for i in range(64):
  r,c=divmod(i,8)
  if s.h>>i&1:x[5,r,c]=x[5,r,c+1]=1
  if s.v>>i&1:x[6,r,c]=x[6,r+1,c]=1
 x[7].fill(s.ply/200);x[8].fill(1-s.player);x[9,0]=1;x[10,8]=1
 return x
def encode_batch(states):return np.stack([encode(s) for s in states])

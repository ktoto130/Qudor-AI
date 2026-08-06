import torch
from torch import nn
from .core.engine import ACTION_SIZE
from .core.encoding import PLANES
class ResBlock(nn.Module):
 def __init__(self,c):super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1,bias=False),nn.BatchNorm2d(c))
 def forward(self,x):return torch.nn.functional.silu(x+self.net(x))
class PolicyValueNet(nn.Module):
 def __init__(self,channels=64,blocks=6):
  super().__init__();self.channels=channels;self.blocks=blocks
  self.stem=nn.Sequential(nn.Conv2d(PLANES,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.SiLU())
  self.body=nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
  self.policy=nn.Sequential(nn.Conv2d(channels,8,1),nn.SiLU(),nn.Flatten(),nn.Linear(8*81,ACTION_SIZE))
  self.value=nn.Sequential(nn.Conv2d(channels,4,1),nn.SiLU(),nn.Flatten(),nn.Linear(4*81,128),nn.SiLU(),nn.Linear(128,1),nn.Tanh())
 def forward(self,x):z=self.body(self.stem(x));return self.policy(z),self.value(z).squeeze(1)
def masked_policy(logits,masks):return logits.masked_fill(~masks,-1e9)

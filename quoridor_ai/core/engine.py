from __future__ import annotations
from dataclasses import dataclass
from collections import deque
N=9; ACTION_SIZE=209
@dataclass(slots=True)
class State:
    p0:int=76; p1:int=4; walls0:int=10; walls1:int=10; h:int=0; v:int=0; player:int=0; ply:int=0
    def copy(self): return State(self.p0,self.p1,self.walls0,self.walls1,self.h,self.v,self.player,self.ply)
    @property
    def winner(self):
        if self.p0//9==0:return 0
        if self.p1//9==8:return 1
        return None

def rc(x):return divmod(x,9)
def bit(r,c):return 1<<(r*8+c)
def blocked(a,b,s):
    ar,ac=rc(a);br,bc=rc(b)
    if ar==br:
        c=min(ac,bc); return ((ar>0 and s.v&bit(ar-1,c)) or (ar<8 and s.v&bit(ar,c)))!=0
    r=min(ar,br); return ((ac>0 and s.h&bit(r,ac-1)) or (ac<8 and s.h&bit(r,ac)))!=0

def neighbors(x,s):
    r,c=rc(x)
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:
            y=rr*9+cc
            if not blocked(x,y,s):yield y

def pawn_moves(s,player=None):
    p=s.player if player is None else player; me=s.p0 if p==0 else s.p1; opp=s.p1 if p==0 else s.p0; out=set()
    mr,mc=rc(me);or_,oc=rc(opp)
    for q in neighbors(me,s):
        if q!=opp:out.add(q);continue
        dr,dc=or_-mr,oc-mc;rr,cc=or_+dr,oc+dc
        if 0<=rr<9 and 0<=cc<9 and not blocked(opp,rr*9+cc,s):out.add(rr*9+cc)
        else:
            for tr,tc in ((dc,dr),(-dc,-dr)):
                rr,cc=or_+tr,oc+tc
                if 0<=rr<9 and 0<=cc<9 and not blocked(opp,rr*9+cc,s):out.add(rr*9+cc)
    return sorted(out)

def has_path(s,player):
    start=s.p0 if player==0 else s.p1;goal=0 if player==0 else 8;q=deque([start]);seen=1<<start
    while q:
        x=q.popleft()
        if x//9==goal:return True
        for y in neighbors(x,s):
            if not(seen>>y)&1:seen|=1<<y;q.append(y)
    return False

def can_wall(s,o,r,c):
    if not(0<=r<8 and 0<=c<8) or (s.walls0 if s.player==0 else s.walls1)<=0:return False
    b=bit(r,c)
    if (s.h|s.v)&b:return False
    if o=='h':
        if (c>0 and s.h&bit(r,c-1)) or (c<7 and s.h&bit(r,c+1)):return False
        n=s.copy();n.h|=b
    else:
        if (r>0 and s.v&bit(r-1,c)) or (r<7 and s.v&bit(r+1,c)):return False
        n=s.copy();n.v|=b
    return has_path(n,0) and has_path(n,1)

def legal_actions(s):
    if s.winner is not None:return []
    a=pawn_moves(s)
    if (s.walls0 if s.player==0 else s.walls1):
        for r in range(8):
            for c in range(8):
                if can_wall(s,'h',r,c):a.append(81+r*8+c)
                if can_wall(s,'v',r,c):a.append(145+r*8+c)
    return a

def apply_unchecked(s,a):
    n=s.copy();p=s.player
    if a<81:
        if p==0:n.p0=a
        else:n.p1=a
    elif a<145:
        x=a-81;n.h|=bit(x//8,x%8);n.walls0-=p==0;n.walls1-=p==1
    else:
        x=a-145;n.v|=bit(x//8,x%8);n.walls0-=p==0;n.walls1-=p==1
    n.player=1-p;n.ply+=1;return n

def apply(s,a):
    if a not in legal_actions(s):raise ValueError(a)
    return apply_unchecked(s,a)

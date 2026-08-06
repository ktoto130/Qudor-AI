import torch
from quoridor_ai.model import PolicyValueNet
from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.core.engine import State
def test_shapes():
 m=PolicyValueNet(16,1);p,v=m(torch.from_numpy(encode_batch([State(),State()])));assert p.shape==(2,209) and v.shape==(2,)

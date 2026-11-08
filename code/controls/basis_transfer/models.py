"""Six-descriptor independent-geometry regressors; shared 64-D output head."""
import hashlib
import numpy as np
import torch
from torch import nn

FEATURES=['R_angstrom','n_spatial','n_electrons','n_parameters','HOMO_canonical_Eh','LUMO_canonical_Eh']
# Fixed unit/size rescaling, not estimated from val/test or from target labels.
# No features are discarded; source and target use exactly the same convention.
OFFSET=np.array([1.8,0,0,0,0,0],dtype=np.float64)
SCALE=np.array([1.,20.,4.,640.,1.,1.],dtype=np.float64)

def features(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():
        raise ValueError('Expected finite [N,6] source descriptors')
    return ((a-OFFSET)/SCALE).astype(np.float32)

class TransformerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_w=nn.Parameter(torch.randn(6,64)*0.02)
        self.feature_b=nn.Parameter(torch.zeros(6,64))
        self.readout=nn.Parameter(torch.randn(1,1,64)*0.02)
        # Independent initializations, not cloned identical encoder layers.
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=64,nhead=2,dim_feedforward=256,dropout=.1,
            activation='gelu',batch_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(64)
    def forward(self,x):
        h=x.unsqueeze(-1)*self.feature_w.unsqueeze(0)+self.feature_b.unsqueeze(0)
        h=torch.cat([self.readout.expand(x.shape[0],-1,-1),h],dim=1)
        for layer in self.layers:h=layer(h)
        return self.norm(h[:,0,:])

class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,283),nn.GELU(),nn.Dropout(.1),
              nn.Linear(283,283),nn.GELU(),nn.Dropout(.1),
              nn.Linear(283,64),nn.GELU(),nn.Dropout(.1))
    def forward(self,x):return self.net(x)

class Model(nn.Module):
    def __init__(self,family,nout):
        super().__init__()
        if family not in ('transformer','mlp') or nout not in (44,434,629):
            raise ValueError('Unsupported architecture/output space')
        self.backbone=TransformerBackbone() if family=='transformer' else MLPBackbone()
        self.head=nn.Linear(64,nout)
    def forward(self,x):return self.head(self.backbone(x))

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def initialize(family,nout,seed,source_state=None):
    # Same random model and head for scratch and finetuning at each seed/output size.
    torch.manual_seed(15000+int(seed))
    m=Model(family,nout)
    before=state_hash(m.head.state_dict())
    if source_state is not None:
        bb={k[len('backbone.'):]:v for k,v in source_state.items() if k.startswith('backbone.')}
        m.backbone.load_state_dict(bb,strict=True)
        if state_hash(m.head.state_dict())!=before:raise RuntimeError('Target head was inadvertently transferred')
    return m

@torch.no_grad()
def predict(model,x,batch=64):
    model.eval()
    t=torch.as_tensor(np.asarray(x),dtype=torch.float32)
    return torch.cat([model(t[i:i+batch]) for i in range(0,len(t),batch)]).cpu().numpy()

def invariance(model,x,context):
    ref=predict(model,x,len(x));order=np.random.RandomState(2025).permutation(len(x))
    cases={'single':predict(model,x,1),'batch16':predict(model,x,16),
           'permutation':predict(model,np.asarray(x)[order],len(x))[np.argsort(order)]}
    n=min(16,len(context))
    cases['extra_context']=predict(model,np.concatenate([context[:n],x,context[:n]]),len(x)+2*n)[n:n+len(x)]
    rows=[]
    for name,v in cases.items():
        d=np.abs(v-ref);passed=bool(np.all(d<=1e-5+1e-5*np.abs(ref)))
        rows.append({'case':name,'max_abs':float(d.max()),'passed':passed})
    return {'atol':1e-5,'rtol':1e-5,'passed':all(x['passed'] for x in rows),'cases':rows}

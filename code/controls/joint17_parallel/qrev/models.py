"""New architectures. No legacy checkpoint is reused.

Transformer: 14 scalar feature tokens + one learned readout token; attention is
strictly internal to each geometry. MLP receives exactly the same 14 numbers.
"""
import torch
from torch import nn

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, heads=2, layers=2,
                 ff_dim=256, dropout=0.1):
        super().__init__()
        if d_model % heads or layers < 1 or input_dim < 2:
            raise ValueError('Invalid Transformer dimensions')
        self.input_dim = input_dim
        self.feature_weight = nn.Parameter(torch.empty(input_dim,d_model))
        self.feature_bias = nn.Parameter(torch.empty(input_dim,d_model))
        self.readout = nn.Parameter(torch.empty(1,1,d_model))
        # Create layers independently instead of cloning identical initialized weights.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d_model,nhead=heads,dim_feedforward=ff_dim,dropout=dropout,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,output_dim)
        nn.init.normal_(self.feature_weight,std=0.02)
        nn.init.normal_(self.feature_bias,std=0.02)
        nn.init.normal_(self.readout,std=0.02)
        # Small but nonconstant initial outputs; no label-dependent initialization.
        nn.init.normal_(self.head.weight,std=0.02)
        nn.init.zeros_(self.head.bias)

    def tokens(self,x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('Expected [B,%d] features, got %s' % (self.input_dim,tuple(x.shape)))
        feature_tokens = x.unsqueeze(-1)*self.feature_weight.unsqueeze(0)+self.feature_bias.unsqueeze(0)
        return torch.cat([self.readout.expand(x.shape[0],-1,-1),feature_tokens],dim=1)

    def forward(self,x):
        h = self.tokens(x)  # [B,15,d]; B is NEVER an attention sequence axis.
        for layer in self.layers:
            h = layer(h)
        return self.head(self.final_norm(h[:,0,:]))

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden=(224,224),dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        blocks = []
        previous = input_dim
        for width in hidden:
            if width < 1:
                raise ValueError('MLP widths must be positive')
            blocks.extend([nn.Linear(previous,width),nn.GELU(),nn.Dropout(dropout)])
            previous = width
        blocks.append(nn.Linear(previous,output_dim))
        self.net = nn.Sequential(*blocks)
        nn.init.normal_(self.net[-1].weight,std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('Invalid MLP input shape')
        return self.net(x)

def make_model(config,input_dim,output_dim):
    model = config['model']
    if model == 'transformer':
        return FeatureTransformer(input_dim,output_dim,**config['architecture'])
    if model == 'mlp':
        return MLP(input_dim,output_dim,**config['architecture'])
    if model == 'lih_gnn':
        return LiHGraphNetwork(input_dim,output_dim,**config['architecture'])
    raise ValueError('Unknown model: '+model)

class BondInteraction(nn.Module):
    """Continuous scalar-distance filter on the two directed Li--H edges."""
    def __init__(self, width, n_rbf, dropout):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(n_rbf,width),nn.GELU(),nn.Linear(width,width))
        self.neighbor = nn.Linear(width,width,bias=False)
        self.update = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),
                                    nn.Linear(width,width))
        self.norm = nn.LayerNorm(width)

    def forward(self,h,radial):
        # Each geometry contains exactly two atoms. No edge ever crosses B.
        neighbor = self.neighbor(h.flip(1))
        message = neighbor*self.filter(radial).unsqueeze(1)
        return self.norm(h+self.update(message))

class LiHGraphNetwork(nn.Module):
    """SchNet-inspired *parameter-output* MPNN for a fixed LiH diatomic only.

    Node identities are Li and H; two directed edges share the bond distance.
    R is the same standardized bondlength feature seen by both other models.
    Species/topology are fixed constants on this representative dataset; no
    unknown polyatomic structure or label-derived graph is fabricated.
    Not a full reproduction of SchNet, not an energy network, not a 17-task GNN.
    """
    def __init__(self,input_dim,output_dim,width=64,interactions=3,n_rbf=16,
                 head_hidden=128,dropout=0.1):
        super().__init__()
        self.input_dim=input_dim
        if input_dim!=14 or interactions<1 or n_rbf<2:
            raise ValueError('This LiH controlled baseline expects the original 14 inputs')
        self.atom_embedding=nn.Embedding(4,width)  # atomic numbers 3 (Li), 1 (H)
        self.register_buffer('atomic_numbers',torch.tensor([3,1],dtype=torch.long))
        self.register_buffer('rbf_centers',torch.linspace(-2.5,2.5,n_rbf))
        self.gamma=(float(n_rbf-1)/5.0)**2
        self.interactions=nn.ModuleList([BondInteraction(width,n_rbf,dropout)
                                         for _ in range(interactions)])
        self.head=nn.Sequential(nn.Linear(width+input_dim,head_hidden),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(head_hidden,output_dim))
        nn.init.normal_(self.head[-1].weight,std=0.02);nn.init.zeros_(self.head[-1].bias)

    def graph_embedding(self,x,node_order=None):
        if x.ndim!=2 or x.shape[1]!=self.input_dim:
            raise ValueError('Expected [B,14] LiH features')
        # Standardization is fitted on LiH TRAIN rows, shared by all networks.
        r=x[:,0:1]
        radial=torch.exp(-self.gamma*(r-self.rbf_centers.unsqueeze(0))**2)
        z=self.atomic_numbers if node_order is None else self.atomic_numbers[node_order]
        h=self.atom_embedding(z).unsqueeze(0).expand(x.shape[0],-1,-1)
        for block in self.interactions:
            h=block(h,radial)
        return h.mean(dim=1)

    def forward(self,x):
        return self.head(torch.cat([self.graph_embedding(x),x],dim=1))

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
    raise ValueError('This v1 implements only transformer and mlp, not a fabricated molecular GNN')

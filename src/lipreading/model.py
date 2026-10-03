"""Dense 3D spatial-temporal features with length-aware recurrent aggregation."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


@dataclass
class ModelConfig:
    classes:int=6
    width:int=8
    growth:int=4
    hidden:int=24
    recurrent_layers:int=2
    aggregation:str="attention"
    dropout:float=.15


class FrameNorm(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.GroupNorm(1,channels)
    def forward(self,x):
        b,c,t,h,w=x.shape
        return self.norm(x.transpose(1,2).reshape(b*t,c,h,w)).reshape(b,t,c,h,w).transpose(1,2)


class DenseLayer(nn.Module):
    def __init__(self,channels,growth):
        super().__init__()
        self.layer=nn.Sequential(FrameNorm(channels),nn.SiLU(),nn.Conv3d(channels,growth,3,padding=1,bias=False))
    def forward(self,x): return torch.cat([x,self.layer(x)],dim=1)


class VisualWordRecognizer(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if config.classes < 2 or config.aggregation not in ['mean','gru','attention']:
            raise ValueError("Invalid model configuration")
        w,g=config.width,config.growth
        self.stem=nn.Sequential(nn.Conv3d(1,w,(3,5,5),stride=(1,2,2),padding=(1,2,2),bias=False),nn.SiLU(),nn.AvgPool3d((1,2,2)),DenseLayer(w,g),DenseLayer(w+g,g),nn.SiLU(),nn.AvgPool3d((1,2,2)))
        channels=w+2*g
        if config.aggregation != 'mean':
            self.temporal=nn.GRU(channels,config.hidden,num_layers=config.recurrent_layers,batch_first=True,bidirectional=True,dropout=config.dropout if config.recurrent_layers>1 else 0)
            channels=2*config.hidden
        self.attention=nn.Sequential(nn.Linear(channels,config.hidden),nn.Tanh(),nn.Linear(config.hidden,1)) if config.aggregation=='attention' else None
        self.head=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(channels,config.classes))

    def forward(self,x,lengths,return_attention=False):
        if x.ndim!=5 or x.shape[1]!=1 or lengths.shape!=(len(x),): raise ValueError("Expected B,1,T,H,W and B lengths")
        if torch.any(lengths<1) or torch.any(lengths>x.shape[2]): raise ValueError("Invalid sequence lengths")
        mask=torch.arange(x.shape[2],device=x.device)[None,:] < lengths[:,None].to(x.device)
        x=x*mask[:,None,:,None,None]
        # Spatial-only normalization avoids padded time steps altering valid features.
        for layer in self.stem:
            x=layer(x)*mask[:,None,:,None,None]
        features=x.mean(dim=(-1,-2)).transpose(1,2)
        weights=mask.to(features.dtype)/lengths[:,None].to(x.device)
        if self.config.aggregation != 'mean':
            packed=pack_padded_sequence(features,lengths.cpu(),batch_first=True,enforce_sorted=False)
            encoded,hidden=self.temporal(packed)
            features,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.shape[2])
            if self.attention is not None:
                scores=self.attention(features).squeeze(-1).masked_fill(~mask,float('-inf'))
                weights=scores.softmax(dim=-1)
                pooled=(features*weights[:,:,None]).sum(1)
            else:
                pooled=torch.cat([hidden[-2],hidden[-1]],dim=-1)
        else: pooled=(features*weights[:,:,None]).sum(1)
        logits=self.head(pooled)
        return (logits,weights) if return_attention else logits

    def parameter_count(self): return sum(p.numel() for p in self.parameters())


def save_bundle(path,model,labels,preprocessing,temperature=1.,metadata=None):
    if len(labels)!=model.config.classes or len(set(labels))!=len(labels): raise ValueError("Invalid vocabulary")
    kind='thesis' if hasattr(model.config,'architecture') else 'compact'
    torch.save(dict(format_version=1,model_kind=kind,config=asdict(model.config),state_dict=model.state_dict(),labels=labels,preprocessing=preprocessing,temperature=float(temperature),metadata=metadata or {}),path)


def load_bundle(path,device='cpu'):
    bundle=torch.load(path,map_location=device,weights_only=True)
    if bundle.get('format_version')!=1: raise ValueError("Unsupported bundle format")
    if bundle.get('model_kind','compact')=='thesis':
        from .thesis_model import ThesisConfig,ThesisRecognizer
        config=ThesisConfig(**bundle['config']);model_class=ThesisRecognizer
    else:config=ModelConfig(**bundle['config']);model_class=VisualWordRecognizer
    labels=bundle['labels']
    if len(labels)!=config.classes or len(set(labels))!=len(labels):raise ValueError("Bundle vocabulary mismatch")
    if not .2<=bundle['temperature']<=10:raise ValueError("Invalid calibration temperature")
    model=model_class(config).to(device);model.load_state_dict(bundle['state_dict'],strict=True);model.eval()
    return model,bundle

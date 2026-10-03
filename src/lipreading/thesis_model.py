"""Configurable 3D DenseNet–LipNet feature fusion for thesis-scale experiments.

This implementation follows the documented architectural principles. Checkpoints
from the original prototype are not assumed to be numerically interchangeable.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .model import DenseLayer,FrameNorm


@dataclass
class ThesisConfig:
    classes:int=1000
    input_channels:int=1
    initial_features:int=64
    growth:int=32
    block_layers:tuple=(4,8,12,8)
    lip_channels:tuple=(64,128,256)
    fusion_features:int=256
    hidden:int=64
    recurrent_layers:int=2
    dropout:float=.2
    architecture:str='fusion'
    recurrent:str='gru'
    pooling:str='attention'


def masked_apply(layer,x,mask):
    return layer(x)*mask[:,None,:,None,None]


class DenseEncoder(nn.Module):
    def __init__(self,config,temporal=True):
        super().__init__();self.layers=nn.ModuleList()
        w=config.initial_features;kt=3 if temporal else 1
        self.layers.extend([nn.Conv3d(config.input_channels,w,(kt,7,7),stride=(1,2,2),padding=(kt//2,3,3),bias=False),FrameNorm(w),nn.SiLU(),nn.AvgPool3d((1,2,2))])
        for index,count in enumerate(config.block_layers):
            for _ in range(count):
                layer=DenseLayer(w,config.growth)
                if not temporal:layer.layer[-1]=nn.Conv3d(w,config.growth,(1,3,3),padding=(0,1,1),bias=False)
                self.layers.append(layer);w+=config.growth
            if index!=len(config.block_layers)-1:
                self.layers.extend([FrameNorm(w),nn.SiLU(),nn.Conv3d(w,w//2,1,bias=False),nn.AvgPool3d((1,2,2))]);w//=2
        self.layers.extend([FrameNorm(w),nn.SiLU()]);self.channels=w

    def forward(self,x,mask):
        for layer in self.layers:x=masked_apply(layer,x,mask)
        return x


class LipNetEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.layers=nn.ModuleList();channels=config.input_channels
        for width in config.lip_channels:
            self.layers.extend([nn.Conv3d(channels,width,(3,3,3),padding=1,bias=False),FrameNorm(width),nn.SiLU(),nn.MaxPool3d((1,2,2)),nn.Dropout3d(config.dropout)])
            channels=width
        self.channels=channels

    def forward(self,x,mask):
        for layer in self.layers:x=masked_apply(layer,x,mask)
        return x


class ThesisRecognizer(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if config.architecture not in ['fusion','dense3d','dense2d','lipnet'] or config.recurrent not in ['gru','lstm'] or config.pooling not in ['last','attention']:
            raise ValueError('Invalid thesis model configuration')
        channels=0
        if config.architecture in ['fusion','dense3d','dense2d']:
            self.dense=DenseEncoder(config,temporal=config.architecture!='dense2d');channels+=self.dense.channels
        if config.architecture in ['fusion','lipnet']:
            self.lip=LipNetEncoder(config);channels+=self.lip.channels
        # Concatenate aligned feature volumes BEFORE global spatial pooling.
        self.fusion=nn.Conv3d(channels,config.fusion_features,1,bias=False)
        recurrent=nn.GRU if config.recurrent=='gru' else nn.LSTM
        self.temporal=recurrent(config.fusion_features,config.hidden,num_layers=config.recurrent_layers,bidirectional=True,batch_first=True,dropout=config.dropout if config.recurrent_layers>1 else 0)
        self.attention=nn.Sequential(nn.Linear(2*config.hidden,config.hidden),nn.Tanh(),nn.Linear(config.hidden,1))
        self.head=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(2*config.hidden,config.classes))

    def forward(self,x,lengths,return_attention=False):
        if x.ndim!=5 or x.shape[1]!=self.config.input_channels or lengths.shape!=(len(x),):raise ValueError('Invalid input dimensions')
        if torch.any(lengths<1) or torch.any(lengths>x.shape[2]):raise ValueError('Invalid lengths')
        if min(x.shape[-2:])<2**(len(self.config.block_layers)+1) and hasattr(self,'dense'):raise ValueError('Spatial resolution too small for dense transitions')
        mask=torch.arange(x.shape[2],device=x.device)[None,:]<lengths[:,None].to(x.device);x=x*mask[:,None,:,None,None]
        branches=[]
        if hasattr(self,'dense'):branches.append(self.dense(x,mask))
        if hasattr(self,'lip'):
            lip=self.lip(x,mask)
            if branches:lip=nn.functional.adaptive_avg_pool3d(lip,branches[0].shape[2:])
            branches.append(lip)
        fused=self.fusion(torch.cat(branches,1))*mask[:,None,:,None,None]
        sequence=fused.mean((-1,-2)).transpose(1,2)
        packed=pack_padded_sequence(sequence,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,state=self.temporal(packed);features,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.shape[2])
        if isinstance(state,tuple):state=state[0]
        if self.config.pooling=='attention':
            weights=self.attention(features).squeeze(-1).masked_fill(~mask,float('-inf')).softmax(-1);pooled=(features*weights[:,:,None]).sum(1)
        else:
            pooled=torch.cat([state[-2],state[-1]],1);weights=mask.float()/lengths[:,None].to(x.device)
        logits=self.head(pooled)
        return (logits,weights) if return_attention else logits

    def parameter_count(self):return sum(p.numel() for p in self.parameters())


def compact_thesis_config(classes=6,architecture='fusion'):
    return ThesisConfig(classes=classes,initial_features=8,growth=4,block_layers=(2,2),lip_channels=(8,12,16),fusion_features=24,hidden=24,architecture=architecture)

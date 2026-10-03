"""Closed-set metrics, calibration and source-clip bootstrap intervals."""
import numpy as np
import torch


def classification_metrics(logits,targets,temperature=1.,bins=10):
    z=np.asarray(logits,dtype=np.float64)/temperature;y=np.asarray(targets,dtype=np.int64)
    if z.ndim!=2 or len(z)!=len(y) or not len(y) or not np.isfinite(z).all():raise ValueError("Invalid logits/targets")
    if np.any(y<0) or np.any(y>=z.shape[1]):raise ValueError("Target outside vocabulary")
    z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);pred=p.argmax(1)
    confusion=np.zeros((z.shape[1],z.shape[1]),dtype=int);np.add.at(confusion,(y,pred),1)
    tp=confusion.diagonal();support=confusion.sum(1);predsupport=confusion.sum(0)
    recall=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predsupport,out=np.zeros_like(tp,dtype=float),where=(support+predsupport)>0)
    present=support>0;confidence=p.max(1);correct=pred==y;ece=0.
    for lower,upper in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        selected=(confidence>lower)&(confidence<=upper)
        if selected.any():ece+=selected.mean()*abs(confidence[selected].mean()-correct[selected].mean())
    k=min(3,z.shape[1]);top=np.argsort(p,axis=1)[:,-k:]
    return dict(n=len(y),accuracy=float(correct.mean()),balanced_accuracy=float(recall[present].mean()),macro_f1=float(f1.mean()),top3_accuracy=float((top==y[:,None]).any(1).mean()),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),ece=float(ece),brier=float(((p-np.eye(z.shape[1])[y])**2).sum(1).mean()),confusion=confusion.tolist(),class_support=support.tolist(),per_class_recall=recall.tolist())


def fit_temperature(logits,targets):
    z=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(targets,dtype=torch.long)
    log_temp=torch.zeros((),dtype=torch.float64,requires_grad=True);optimizer=torch.optim.Adam([log_temp],lr=.03)
    for _ in range(150):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(z/log_temp.exp().clamp(.2,10),y);loss.backward();optimizer.step()
        with torch.no_grad():log_temp.clamp_(np.log(.2),np.log(10))
    return float(log_temp.exp().detach())


def group_bootstrap(logits,targets,groups,replicates=500,seed=2026):
    groups=np.asarray(groups);unique=np.unique(groups);rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        sampled=rng.choice(unique,len(unique),replace=True)
        index=np.concatenate([np.flatnonzero(groups==group) for group in sampled])
        values.append(classification_metrics(np.asarray(logits)[index],np.asarray(targets)[index])['accuracy'])
    return dict(accuracy_ci95=np.quantile(values,[.025,.975]).tolist(),resampling_unit="source_clip",groups=len(unique),replicates=replicates,seed=seed)

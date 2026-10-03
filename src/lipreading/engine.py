import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CachedDataset
from .metrics import classification_metrics,fit_temperature,group_bootstrap
from .model import ModelConfig,VisualWordRecognizer,save_bundle,load_bundle


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def collect_predictions(model,dataset,device='cpu',batch_size=32):
    logits=[];targets=[];model.eval()
    with torch.inference_mode():
        for x,lengths,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            logits.append(model(x.to(device),lengths.to(device)).cpu().numpy());targets.append(y.numpy())
    return np.concatenate(logits),np.concatenate(targets)


def train(cache,output,aggregation='attention',epochs=12,seed=17,batch_size=32,lr=.003,augment=False,device='cpu',architecture=None,model_config=None):
    if epochs<1:raise ValueError("epochs must be positive")
    seed_everything(seed);output=Path(output);output.mkdir(parents=True,exist_ok=True);cache=Path(cache)
    manifest=json.loads((cache/'manifest.json').read_text(encoding='utf8'))
    train_data=CachedDataset(cache/'train.npz',augment);val_data=CachedDataset(cache/'validation.npz');test_data=CachedDataset(cache/'test.npz')
    if architecture or model_config:
        from .thesis_model import ThesisConfig,ThesisRecognizer,compact_thesis_config
        config=ThesisConfig(**model_config) if model_config else compact_thesis_config(len(manifest['labels']),architecture)
        if config.classes!=len(manifest['labels']):raise ValueError('Model class count must match the prepared training vocabulary')
        model=ThesisRecognizer(config).to(device)
    else:
        config=ModelConfig(classes=len(manifest['labels']),aggregation=aggregation)
        model=VisualWordRecognizer(config).to(device)
    counts=torch.bincount(train_data.y,minlength=config.classes).float()
    if torch.any(counts==0):raise ValueError("Training class has no accepted samples")
    weights=counts.sum()/counts;weights=weights/weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,generator=generator)
    best=-1.;best_epoch=0;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for x,lengths,y in loader:
            optimizer.zero_grad();logits=model(x.to(device),lengths.to(device));loss=criterion(logits,y.to(device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        scheduler.step();val_logits,val_y=collect_predictions(model,val_data,device,batch_size)
        metrics=classification_metrics(val_logits,val_y)
        row=dict(epoch=epoch,train_weighted_loss=float(np.mean(losses)),validation_accuracy=metrics['accuracy'],validation_balanced_accuracy=metrics['balanced_accuracy'],validation_macro_f1=metrics['macro_f1'])
        history.append(row)
        if metrics['balanced_accuracy']>best:
            best=metrics['balanced_accuracy'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        print(f"{aggregation} seed={seed} epoch={epoch}/{epochs} loss={row['train_weighted_loss']:.4f} val_balanced={best:.4f}",flush=True)
    model.load_state_dict(best_state);model.eval()
    val_logits,val_y=collect_predictions(model,val_data,device,batch_size)
    temperature=fit_temperature(val_logits,val_y)
    test_logits,test_y=collect_predictions(model,test_data,device,batch_size)
    test_metrics=classification_metrics(test_logits,test_y,temperature)
    test_metrics.update(group_bootstrap(test_logits,test_y,test_data.groups))
    elapsed=time.perf_counter()-start
    metadata=dict(seed=seed,best_epoch=best_epoch,epochs=epochs,batch_size=batch_size,lr=lr,augmentation=augment,architecture=architecture or 'compact',class_weighting="inverse_training_frequency",selection_metric="validation_balanced_accuracy",dataset_source_hashes=manifest['source_hashes'])
    save_bundle(output/'model.pt',model,manifest['labels'],manifest['config'],temperature,metadata)
    reloaded,bundle=load_bundle(output/'model.pt',device)
    x,lengths,_=test_data[0];x=x[None].to(device);lengths=lengths.reshape(1).to(device)
    with torch.inference_mode():difference=float((model(x,lengths)-reloaded(x,lengths)).abs().max())
    if difference>1e-6:raise RuntimeError("Reload changed predictions")
    result=dict(name=aggregation,seed=seed,config=asdict(config),training=metadata,parameters=model.parameter_count(),temperature=temperature,elapsed_seconds=elapsed,history=history,validation=classification_metrics(val_logits,val_y,temperature),test=test_metrics,test_uncalibrated=classification_metrics(test_logits,test_y),reload_max_abs_difference=difference,environment=dict(torch=torch.__version__,numpy=np.__version__,device=device,threads=torch.get_num_threads()))
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(output/'predictions.npz',logits=test_logits,targets=test_y,groups=np.array(test_data.groups))
    return result


def benchmark(bundle_path,frames=12,size=48,repeats=30,device='cpu'):
    model,_=load_bundle(bundle_path,device);x=torch.zeros(1,1,frames,size,size,device=device);lengths=torch.tensor([frames],device=device)
    with torch.inference_mode():
        for _ in range(5):model(x,lengths)
        samples=[]
        for _ in range(repeats):
            if device.startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter();model(x,lengths)
            if device.startswith('cuda'):torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
    return dict(batch=1,frames=frames,size=size,repeats=repeats,p50_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),includes="model forward only; excludes video decoding and preprocessing",device=device,threads=torch.get_num_threads())

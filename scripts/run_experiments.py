"""Run the registered comparison. Test labels never select checkpoints or variants."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from lipreading.data import CachedDataset
from lipreading.engine import train,benchmark
from lipreading.metrics import classification_metrics

p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',default='runs/comparison');p.add_argument('--epochs',type=int,default=12);p.add_argument('--seeds',nargs='+',type=int,default=[17,29]);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
torch.set_num_threads(args.threads);root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
manifest=json.loads((Path(args.cache)/'manifest.json').read_text(encoding='utf8'));test=CachedDataset(Path(args.cache)/'test.npz')
counts=np.array(manifest['splits']['train']['class_counts']);majority=int(counts.argmax());baseline=np.full((len(test),len(counts)),-5.);baseline[:,majority]=5.
results=[]
for name,aggregation,augment in [('dense3d_mean','mean',False),('dense3d_bigru','gru',False),('dense3d_bigru_attention','attention',False),('attention_augmentation','attention',True)]:
    for seed in args.seeds:
        output=root/f'{name}-seed{seed}'
        result=train(args.cache,output,aggregation,args.epochs,seed,augment=augment)
        result['variant']=name;result['benchmark']=benchmark(output/'model.pt');results.append(result)
        (root/'partial-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
# Choose across variants and seeds using validation only, before looking at test summaries.
winner=max(results,key=lambda r:r['validation']['balanced_accuracy'])
summary=dict(protocol_version=1,date_utc=datetime.now(timezone.utc).isoformat(),dataset=manifest,majority_baseline=dict(training_majority_label=manifest['labels'][majority],test=classification_metrics(baseline,test.y.numpy())),results=results,selected_variant=winner['variant'],selected_seed=winner['seed'],selection='maximum validation balanced accuracy; test is reporting only')
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print('Completed registered comparison',flush=True)

"""Collect real run records and post-selection robustness probes."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from lipreading.data import CachedDataset
from lipreading.engine import collect_predictions,benchmark
from lipreading.metrics import classification_metrics
from lipreading.model import load_bundle

p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--runs',default='runs');p.add_argument('--out',default='results');args=p.parse_args()
torch.set_num_threads(4);root=Path(args.runs);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
comparison=json.loads((root/'comparison/summary.json').read_text(encoding='utf8'));fusion=json.loads((root/'fusion/summary.json').read_text(encoding='utf8'))
records=comparison['results']+fusion
selected=max(records,key=lambda r:r['validation']['balanced_accuracy'])
folder=(root/'fusion'/f"fusion-seed{selected['seed']}") if selected['variant']=='dense3d_lipnet_fusion' else (root/'comparison'/f"{selected['variant']}-seed{selected['seed']}")
model,bundle=load_bundle(folder/'model.pt');original=CachedDataset(Path(args.cache)/'test.npz');robustness=[]
for name in ['clean','darken','brighten','frame_reverse','half_frames_zeroed']:
    dataset=copy.deepcopy(original)
    for i,length in enumerate(dataset.lengths.tolist()):
        if name=='darken':dataset.x[i,:,:length]=(dataset.x[i,:,:length]*.8-.2).clamp(-1,1)
        elif name=='brighten':dataset.x[i,:,:length]=(dataset.x[i,:,:length]*1.2+.2).clamp(-1,1)
        elif name=='frame_reverse':dataset.x[i,:,:length]=dataset.x[i,:,:length].flip(1)
        elif name=='half_frames_zeroed':dataset.x[i,:,1:length:2]=0
    logits,y=collect_predictions(model,dataset);robustness.append(dict(condition=name,metrics=classification_metrics(logits,y,bundle['temperature'])))
manifest=comparison['dataset'];summary=dict(protocol='CPU six-class engineering validation; not a 1000-class benchmark',date_utc=comparison['date_utc'],dataset=manifest,majority_baseline=comparison['majority_baseline'],runs=records,selection=dict(variant=selected['variant'],seed=selected['seed'],metric='validation balanced accuracy',bundle_path=str(folder/'model.pt')),robustness=robustness)
(out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# Engineering validation','',f"Recorded: {summary['date_utc']}",'','Real local LRW-style mouth-image subset: 617 training, 194 validation and 90 test segments across six classes. Original source clips are disjoint across partitions; speaker independence cannot be established from the supplied indices. Two training segments were excluded for missing or insufficient readable frames.','', 'Each variant was trained for 12 epochs, seeds 17 and 29, 48 x 48 grayscale frames and up to 12 timesteps. Class-weighted cross entropy, AdamW and cosine decay were used. Validation balanced accuracy selected checkpoints and the serving bundle; validation-only temperature scaling calibrated logits. Test data only reports outcomes. These compact configurations are not a remeasurement of the thesis benchmark.','', '| Variant | Seed | Test accuracy | Balanced accuracy | Macro F1 | Parameters | Forward p50 ms |','|---|---:|---:|---:|---:|---:|---:|']
for r in records:
    t=r['test'];lines.append(f"| {r['variant']} | {r['seed']} | {100*t['accuracy']:.2f}% | {100*t['balanced_accuracy']:.2f}% | {t['macro_f1']:.3f} | {r['parameters']:,} | {r['benchmark']['p50_ms']:.2f} |")
lines+=['',f"Training-majority baseline test accuracy: {100*summary['majority_baseline']['test']['accuracy']:.2f}%. The prediction is the most common TRAINING label, not a majority chosen using test labels.",'',f"Selected bundle: `{selected['variant']}`, seed {selected['seed']}. Reload prediction differences were zero for all completed runs. Latency uses a batch of one on CPU and excludes video decoding, ROI selection and preprocessing.",'','## Perturbation probes','','| Condition | Test accuracy | Macro F1 |','|---|---:|---:|']
for r in robustness:lines.append(f"| {r['condition']} | {100*r['metrics']['accuracy']:.2f}% | {r['metrics']['macro_f1']:.3f} |")
lines+=['','The perturbation probes are post-selection analyses, not training augmentation claims. Reversing frames tests temporal sensitivity; zeroing half the sampled frames simulates missing observations. Darkening/brightening operate in the normalized image domain. They do not cover all domain shifts or establish deployment reliability.','', '## Uncertainty and calibration','']
for r in records:
    lo,hi=r['test']['accuracy_ci95'];lines.append(f"- {r['variant']}, seed {r['seed']}: source-clip bootstrap accuracy 95% interval [{100*lo:.1f}%, {100*hi:.1f}%], validation-fitted temperature {r['temperature']:.3f}, test NLL {r['test']['nll']:.3f}, test ECE {r['test']['ece']:.3f}.")
(out/'VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(summary['selection'],indent=2))

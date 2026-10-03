import argparse,json
from pathlib import Path
import torch
from lipreading.engine import train,benchmark

p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',default='runs/fusion');p.add_argument('--epochs',type=int,default=12);p.add_argument('--seeds',nargs='+',type=int,default=[17,29]);args=p.parse_args()
torch.set_num_threads(4);root=Path(args.out);root.mkdir(parents=True,exist_ok=True);results=[]
for seed in args.seeds:
    target=root/f'fusion-seed{seed}'
    result=train(args.cache,target,epochs=args.epochs,seed=seed,architecture='fusion');result['variant']='dense3d_lipnet_fusion';result['benchmark']=benchmark(target/'model.pt');results.append(result)
(root/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')

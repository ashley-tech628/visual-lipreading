import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from .data import build_cache
from .engine import train,benchmark
from .inference import predict_frames
from .model import load_bundle


def main():
    parser=argparse.ArgumentParser(description="Visual word recognition experiments and inference")
    parser.add_argument('--threads',type=int,default=4)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--info',required=True);p.add_argument('--images',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=48);p.add_argument('--frames',type=int,default=12)
    p=sub.add_parser('train');p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--aggregation',choices=['mean','gru','attention'],default='attention');p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=17);p.add_argument('--augment',action='store_true');p.add_argument('--device',default='cpu')
    p.add_argument('--architecture',choices=['fusion','dense3d','dense2d','lipnet']);p.add_argument('--config',help='JSON thesis model configuration')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.003)
    p=sub.add_parser('predict');p.add_argument('--bundle',required=True);p.add_argument('--frames-dir',required=True);p.add_argument('--roi',nargs=4,type=float)
    p=sub.add_parser('benchmark');p.add_argument('--bundle',required=True)
    p=sub.add_parser('serve');p.add_argument('--bundle',required=True);p.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if args.command=='prepare':result=build_cache(args.info,args.images,args.out,args.size,args.frames)
    elif args.command=='train':result=train(args.cache,args.out,args.aggregation,args.epochs,args.seed,batch_size=args.batch_size,lr=args.lr,augment=args.augment,device=args.device,architecture=args.architecture,model_config=json.loads(Path(args.config).read_text()) if args.config else None)
    elif args.command=='benchmark':result=benchmark(args.bundle)
    elif args.command=='predict':
        model,bundle=load_bundle(args.bundle);paths=sorted(Path(args.frames_dir).glob('*.jpg'),key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem)
        images=[]
        for path in paths:
            with Image.open(path) as image:images.append(image.copy())
        result=predict_frames(model,bundle,images,args.roi)
    else:
        from .server import serve
        serve(args.bundle,args.port);return
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

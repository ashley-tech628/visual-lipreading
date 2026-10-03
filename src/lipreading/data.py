"""LRW-style segment loading with a single vocabulary and auditable exclusions."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def preprocess_frames(frames, size=48, max_frames=12, roi=None):
    if not frames:
        raise ValueError("No readable frames")
    if roi is not None:
        if len(roi) != 4 or not all(math.isfinite(float(v)) for v in roi):
            raise ValueError("ROI must have four finite normalized coordinates")
        x, y, w, h = map(float, roi)
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.00001 and y+h <= 1.00001):
            raise ValueError("ROI must be within the image")
    selected = np.linspace(0, len(frames)-1, min(max_frames, len(frames))).round().astype(int)
    volume = np.zeros((1, max_frames, size, size), dtype=np.float32)
    for t, index in enumerate(selected):
        image = frames[index].convert("L")
        if roi is not None:
            a,b,c,d = roi; width,height = image.size
            image = image.crop((int(a*width),int(b*height),int((a+c)*width),int((b+d)*height)))
        image = ImageOps.fit(image, (size,size), method=Image.Resampling.BILINEAR)
        volume[0,t] = (np.asarray(image,dtype=np.float32)/255.0 - 0.5)/0.5
    return volume, len(selected)


def read_index(path, fps=25):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for n, row in enumerate(csv.reader(file),1):
            if not row: continue
            if len(row) != 5 or not re.fullmatch(r"[A-Za-z0-9_-]+", row[0]):
                raise ValueError(f"Invalid segment record at line {n}")
            start,end = float(row[3]),float(row[4])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f"Invalid segment timing at line {n}")
            rows.append(dict(clip=row[0],word=row[1],label=row[2],start=int(start*fps)+1,end=int(end*fps)+1,line=n))
    if not rows: raise ValueError("Index is empty")
    return rows


def build_cache(info, image_roots, output, size=48, max_frames=12, fps=25, min_coverage=.8):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = {name:read_index(Path(info)/f"{prefix}_1000.txt",fps) for name,prefix in [("train","trn"),("validation","val"),("test","tst")]}
    labels = sorted({r["label"] for r in records["train"]})
    vocab = {label:i for i,label in enumerate(labels)}
    for split, rows in records.items():
        if any(r["label"] not in vocab for r in rows):
            raise ValueError(f"{split} has labels absent from training")
        if len({(r['clip'],r['start'],r['end']) for r in rows}) != len(rows):
            raise ValueError(f"Duplicate segment records in {split}")
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        if {r['clip'] for r in records[a]} & {r['clip'] for r in records[b]}:
            raise ValueError(f"Source clip leakage between {a} and {b}")
    report = dict(labels=labels,config=dict(size=size,max_frames=max_frames,fps=fps,min_coverage=min_coverage),splits={},source_hashes={})
    for split,rows in records.items():
        xs=[]; ys=[]; lengths=[]; groups=[]; words=[]; rejected=[]; coverages=[]
        for row in rows:
            roots = [Path(p)/row['clip'] for p in image_roots]
            root = next((p for p in roots if p.is_dir()),None)
            if root is None:
                rejected.append(dict(line=row['line'],reason="missing_clip_directory")); continue
            requested = list(range(row['start'],row['end']))
            if not requested:
                rejected.append(dict(line=row['line'],reason="empty_frame_interval")); continue
            frames=[]
            for index in requested:
                try:
                    with Image.open(root/f"{index}.jpg") as image: frames.append(image.convert("L").copy())
                except (OSError,ValueError): pass
            coverage = len(frames)/len(requested)
            if coverage < min_coverage:
                rejected.append(dict(line=row['line'],reason="insufficient_readable_frames",coverage=coverage)); continue
            x,length = preprocess_frames(frames,size,max_frames)
            xs.append(x);ys.append(vocab[row['label']]);lengths.append(length);groups.append(row['clip']);words.append(row['word']);coverages.append(coverage)
        if not xs: raise ValueError(f"No readable {split} samples")
        np.savez_compressed(output/f"{split}.npz",x=np.stack(xs),y=np.array(ys,dtype=np.int64),lengths=np.array(lengths,dtype=np.int64),groups=np.array(groups),words=np.array(words))
        report['splits'][split] = dict(index_rows=len(rows),accepted=len(xs),rejected=len(rejected),class_counts=np.bincount(ys,minlength=len(labels)).tolist(),mean_frame_coverage=float(np.mean(coverages)),exclusions=rejected)
        prefix=dict(train='trn',validation='val',test='tst')[split]
        report['source_hashes'][split] = hashlib.sha256((Path(info)/f'{prefix}_1000.txt').read_bytes()).hexdigest()
        print(f"{split}: {len(xs)}/{len(rows)} usable samples",flush=True)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


class CachedDataset(Dataset):
    def __init__(self,path,augment=False):
        with np.load(path,allow_pickle=False) as cache:
            self.x=torch.from_numpy(cache['x'].copy());self.y=torch.from_numpy(cache['y'].copy())
            self.lengths=torch.from_numpy(cache['lengths'].copy());self.groups=cache['groups'].tolist()
        if len(self.x) != len(self.y) or len(self.x) != len(self.groups): raise ValueError("Cache size mismatch")
        self.augment=augment

    def __len__(self): return len(self.y)

    def __getitem__(self,i):
        x=self.x[i].clone()
        if self.augment:
            if torch.rand(()) < .5: x=x.flip(-1)
            x[:,:self.lengths[i]]=(x[:,:self.lengths[i]]*(.9+.2*torch.rand(()))).clamp(-1,1)
        return x,self.lengths[i],self.y[i]

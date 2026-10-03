import numpy as np
import torch
from .data import preprocess_frames


def predict_frames(model,bundle,frames,roi=None,threshold=.5):
    config=bundle['preprocessing'];x,length=preprocess_frames(frames,config['size'],config['max_frames'],roi)
    device=next(model.parameters()).device
    with torch.inference_mode():
        logits,attention=model(torch.from_numpy(x)[None].to(device),torch.tensor([length],device=device),return_attention=True)
        probabilities=(logits/bundle['temperature']).softmax(-1)[0].cpu().numpy()
    order=np.argsort(-probabilities)[:min(3,len(probabilities))]
    return dict(prediction=bundle['labels'][int(order[0])],top3=[dict(label=bundle['labels'][int(i)],probability=float(probabilities[i])) for i in order],low_confidence=bool(probabilities[order[0]]<threshold),threshold=threshold,frames_used=length,temporal_weights=attention[0,:length].cpu().tolist(),note="Closed-set probabilities over the training vocabulary; not an out-of-distribution detector. Temporal weights are not causal explanations.")

"""Landmark-conditioned mouth geometry and temporal ROI stabilization.

Landmarks must be supplied by a separately licensed detector. This module makes
no claim to ship a pretrained 68-point detector.
"""
import numpy as np
from PIL import Image


def mouth_features(landmarks):
    points=np.asarray(landmarks,dtype=np.float64)
    if points.ndim!=3 or points.shape[1:]!=(68,2) or not np.isfinite(points).all():raise ValueError('Expected finite T x 68 x 2 landmarks')
    mouth=points[:,48:68];center=mouth.mean(1,keepdims=True)
    width=np.linalg.norm(points[:,54]-points[:,48],axis=1)
    if np.any(width<=1e-6):raise ValueError('Degenerate mouth landmarks')
    normalized=(mouth-center)/width[:,None,None]
    edges=np.roll(normalized,-1,axis=1)-normalized
    angles=np.arctan2(edges[:,:,1],edges[:,:,0])
    # Sine/cosine avoid the discontinuity at +/- pi.
    return np.concatenate([normalized.reshape(len(points),40),np.sin(angles),np.cos(angles)],axis=1).astype(np.float32)


def smooth_roi_series(rois,window=5,degree=1):
    values=np.asarray(rois,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=4 or not len(values) or not np.isfinite(values).all():raise ValueError('Expected finite T x 4 crop coordinates')
    if window<1 or window%2!=1 or degree<0 or degree>=window:raise ValueError('Invalid polynomial smoothing parameters')
    result=np.zeros_like(values);radius=window//2
    for t in range(len(values)):
        indices=np.arange(max(0,t-radius),min(len(values),t+radius+1));offsets=indices-t
        order=min(degree,len(indices)-1)
        # Local polynomial least squares is the Savitzky-Golay principle.
        design=np.vander(offsets,N=order+1,increasing=True)
        coefficients=np.linalg.lstsq(design,values[indices],rcond=None)[0]
        result[t]=coefficients[0]
    return result


def landmark_rois(landmarks,image_width,image_height,margin=.25,window=5):
    points=np.asarray(landmarks,dtype=float)
    mouth_features(points)  # validates landmark geometry
    mouth=points[:,48:68];low=mouth.min(1);high=mouth.max(1);center=(low+high)/2
    side=np.maximum((high-low).max(1)*(1+2*margin),1.)
    boxes=np.column_stack([center[:,0]-side/2,center[:,1]-side/2,side,side])
    boxes=smooth_roi_series(boxes,window)
    boxes[:,0]=np.clip(boxes[:,0],0,image_width-1);boxes[:,1]=np.clip(boxes[:,1],0,image_height-1)
    boxes[:,2]=np.minimum(np.maximum(boxes[:,2],1),image_width-boxes[:,0]);boxes[:,3]=np.minimum(np.maximum(boxes[:,3],1),image_height-boxes[:,1])
    return boxes/np.array([image_width,image_height,image_width,image_height])

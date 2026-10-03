import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lipreading.data import preprocess_frames,read_index,build_cache
from lipreading.inference import predict_frames
from lipreading.metrics import classification_metrics,fit_temperature,group_bootstrap
from lipreading.model import ModelConfig,VisualWordRecognizer,save_bundle,load_bundle
from lipreading.server import decode_request


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.model=VisualWordRecognizer(ModelConfig(classes=3)).eval()

    def test_padding_content_cannot_change_prediction(self):
        x=torch.randn(2,1,6,24,24);lengths=torch.tensor([3,5]);changed=x.clone();changed[0,:,3:]=100;changed[1,:,5:]=-100
        with torch.no_grad():a=self.model(x,lengths);b=self.model(changed,lengths)
        torch.testing.assert_close(a,b)

    def test_extra_padding_cannot_change_prediction(self):
        x=torch.randn(2,1,5,24,24);lengths=torch.tensor([3,5]);extended=torch.cat([x,torch.randn(2,1,3,24,24)],2)
        with torch.no_grad():a=self.model(x,lengths);b=self.model(extended,lengths)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_attention_masks_and_normalizes(self):
        logits,weights=self.model(torch.randn(2,1,6,24,24),torch.tensor([2,6]),True)
        self.assertTrue(torch.all(weights[0,2:]==0));torch.testing.assert_close(weights.sum(1),torch.ones(2))
        self.assertEqual(logits.shape,(2,3))

    def test_all_ablation_heads_have_gradients(self):
        for aggregation in ['mean','gru','attention']:
            model=VisualWordRecognizer(ModelConfig(classes=3,aggregation=aggregation))
            loss=torch.nn.functional.cross_entropy(model(torch.randn(3,1,5,24,24),torch.tensor([2,4,5])),torch.tensor([0,1,2]))
            loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_lengths_are_rejected(self):
        with self.assertRaises(ValueError):self.model(torch.zeros(1,1,3,24,24),torch.tensor([0]))

    def test_reload_preserves_prediction_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';save_bundle(path,self.model,['a','b','c'],dict(size=24,max_frames=5),2.)
            restored,bundle=load_bundle(path);x=torch.randn(1,1,5,24,24);lengths=torch.tensor([4])
            with torch.no_grad():torch.testing.assert_close(self.model(x,lengths),restored(x,lengths))
            self.assertEqual(bundle['labels'],['a','b','c']);self.assertEqual(bundle['temperature'],2.)

    def test_bundle_rejects_vocabulary_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):save_bundle(Path(tmp)/'x.pt',self.model,['a'],{})

    def test_preprocessing_length_and_zero_padding(self):
        frames=[Image.new('RGB',(30,30),color='white')]*2
        x,length=preprocess_frames(frames,size=24,max_frames=5)
        self.assertEqual(length,2);self.assertTrue(np.all(x[:,2:]==0));self.assertTrue(np.all(x[:,:2]==1))

    def test_crop_outside_image_is_rejected(self):
        with self.assertRaises(ValueError):preprocess_frames([Image.new('RGB',(20,20))],roi=[.9,0,.5,1])

    def test_predictions_include_calibration_and_attention(self):
        bundle=dict(labels=['a','b','c'],temperature=2.,preprocessing=dict(size=24,max_frames=5))
        result=predict_frames(self.model,bundle,[Image.new('RGB',(24,24))]*3)
        self.assertEqual(result['frames_used'],3);self.assertEqual(len(result['temporal_weights']),3)
        self.assertAlmostEqual(sum(r['probability'] for r in result['top3']),1.,places=5)

    def test_metrics_handle_imbalance(self):
        result=classification_metrics([[4,0],[4,0],[4,0],[4,0]],[0,0,0,1])
        self.assertEqual(result['accuracy'],.75);self.assertEqual(result['balanced_accuracy'],.5)

    def test_metrics_reject_invalid_targets(self):
        with self.assertRaises(ValueError):classification_metrics([[1,2]],[2])

    def test_temperature_preserves_ranking(self):
        logits=np.array([[10.,0],[10,0],[0,10],[0,10]]);targets=[0,1,1,1]
        temperature=fit_temperature(logits,targets)
        self.assertLessEqual(classification_metrics(logits,targets,temperature)['nll'],classification_metrics(logits,targets)['nll'])
        np.testing.assert_equal(logits.argmax(1),(logits/temperature).argmax(1))

    def test_bootstrap_uses_source_groups_and_seed(self):
        a=group_bootstrap([[3,0],[3,0],[0,3]],[0,0,0],['a','a','b'],replicates=40,seed=5)
        b=group_bootstrap([[3,0],[3,0],[0,3]],[0,0,0],['a','a','b'],replicates=40,seed=5)
        self.assertEqual(a,b);self.assertEqual(a['groups'],2)

    def test_index_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.txt';path.write_text('../outside,word,label,0,1')
            with self.assertRaises(ValueError):read_index(path)

    def test_source_clip_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ['trn','val','tst']:(root/f'{split}_1000.txt').write_text('same,word,label,0,1')
            with self.assertRaisesRegex(ValueError,'leakage'):build_cache(root,[root],root/'out')

    def test_decode_request_accepts_images_rejects_garbage(self):
        buffer=io.BytesIO();Image.new('RGB',(20,20)).save(buffer,format='PNG')
        frames,roi=decode_request(json.dumps(dict(frames=[base64.b64encode(buffer.getvalue()).decode()])).encode())
        self.assertEqual(len(frames),1)
        with self.assertRaises(ValueError):decode_request(b'{"frames": ["%%%"]}')

if __name__=='__main__':unittest.main()

import unittest
import torch
from lipreading.thesis_model import ThesisRecognizer,compact_thesis_config


class ThesisModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)

    def test_all_backbones_forward_backward(self):
        for name in ['fusion','dense3d','dense2d','lipnet']:
            model=ThesisRecognizer(compact_thesis_config(3,name));x=torch.randn(2,1,5,48,48)
            loss=torch.nn.functional.cross_entropy(model(x,torch.tensor([3,5])),torch.tensor([0,1]));loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_fusion_padding_is_masked(self):
        model=ThesisRecognizer(compact_thesis_config(3)).eval();x=torch.randn(1,1,5,48,48);other=x.clone();other[:,:,3:]=999
        with torch.no_grad():a=model(x,torch.tensor([3]));b=model(other,torch.tensor([3]))
        torch.testing.assert_close(a,b)

    def test_lstm_and_last_state(self):
        config=compact_thesis_config(3);config.recurrent='lstm';config.pooling='last'
        model=ThesisRecognizer(config);logits=model(torch.randn(2,1,5,48,48),torch.tensor([3,5]))
        self.assertEqual(logits.shape,(2,3))

    def test_fusion_has_both_branch_gradients(self):
        model=ThesisRecognizer(compact_thesis_config(3));logits=model(torch.randn(2,1,5,48,48),torch.tensor([3,5]));logits.sum().backward()
        for branch in [model.dense,model.lip]:self.assertTrue(sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None)>0)

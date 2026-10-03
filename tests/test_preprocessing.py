import unittest
import numpy as np
from lipreading.preprocessing import mouth_features,smooth_roi_series,landmark_rois

class GeometryTests(unittest.TestCase):
    def points(self):
        p=np.zeros((4,68,2));theta=np.linspace(0,2*np.pi,20,endpoint=False)
        p[:,48:68,0]=50+15*np.cos(theta);p[:,48:68,1]=50+7*np.sin(theta)
        return p

    def test_features_are_translation_and_scale_invariant(self):
        points=self.points();np.testing.assert_allclose(mouth_features(points),mouth_features(points*2+30),atol=1e-6)

    def test_linear_smoothing_preserves_linear_motion(self):
        rois=np.column_stack([np.arange(7),np.arange(7)*2,np.ones(7)*10,np.ones(7)*10])
        np.testing.assert_allclose(rois,smooth_roi_series(rois),atol=1e-8)

    def test_rois_stay_in_bounds(self):
        rois=landmark_rois(self.points(),100,100)
        self.assertTrue(np.all(rois>=0));self.assertTrue(np.all(rois[:,0]+rois[:,2]<=1));self.assertTrue(np.all(rois[:,1]+rois[:,3]<=1))

    def test_degenerate_landmarks_rejected(self):
        with self.assertRaises(ValueError):mouth_features(np.zeros((2,68,2)))

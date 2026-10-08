"""CPU-only spatial-mask regressions; no checkpoints or dataset required.

Run with: python -m unittest discover -s tests -v
"""
import unittest

import torch

from model.anytop import AnyTop


class SpatialMaskOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1234)
        self.model = AnyTop(
            max_joints=5, feature_len=13, latent_dim=16,
            ff_size=32, num_layers=2, num_heads=2, dropout=0,
            t5_out_dim=8,
        )
        self.model.eval()
        self.motion = torch.randn(2, 5, 13, 4)

    def condition(self, joint_counts):
        batch = len(joint_counts)
        joints_mask = torch.zeros(batch, 1, 1, 6, 6)
        for item, count in enumerate(joint_counts):
            joints_mask[item, :, :, :count + 1, :count + 1] = 1
        return {
            'joints_mask': joints_mask,
            'mask': torch.ones(batch, 1, 1, 5, 5),
            'tpos_first_frame': torch.randn(batch, 5, 13),
            'joints_names_embs': torch.randn(batch, 5, 8),
            'crop_start_ind': torch.zeros(batch, dtype=torch.long),
            'graph_dist': torch.zeros(batch, 5, 5, dtype=torch.long),
            'joints_relations': torch.zeros(batch, 5, 5, dtype=torch.long),
        }

    def assert_batch_independent(self, joint_counts):
        condition = self.condition(joint_counts)
        with torch.no_grad():
            batched = self.model(self.motion, torch.tensor([50, 50]), y=condition)
            for item, count in enumerate(joint_counts):
                single_condition = {
                    key: value[item:item + 1] for key, value in condition.items()
                }
                single = self.model(
                    self.motion[item:item + 1], torch.tensor([50]),
                    y=single_condition,
                )
                torch.testing.assert_close(
                    batched[item, :count], single[0, :count],
                    rtol=1e-5, atol=1e-6,
                )

    def test_different_joint_counts_match_separate_execution(self):
        self.assert_batch_independent([5, 3])

    def test_equal_joint_counts_match_separate_execution(self):
        self.assert_batch_independent([5, 5])

    def test_padded_motion_does_not_change_valid_joint_predictions(self):
        condition = self.condition([5, 3])
        changed = self.motion.clone()
        changed[1, 3:] = torch.randn_like(changed[1, 3:]) * 10
        with torch.no_grad():
            original = self.model(self.motion, torch.tensor([50, 50]), y=condition)
            perturbed = self.model(changed, torch.tensor([50, 50]), y=condition)
        torch.testing.assert_close(
            original[1, :3], perturbed[1, :3], rtol=1e-5, atol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()

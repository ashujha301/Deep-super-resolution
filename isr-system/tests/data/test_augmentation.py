import numpy as np
from src.data.augmentation import PairedAugmenter

def test_manual_augmentation_preserves_corresponding_transform():
    hr = np.arange(4 * 4 * 3).reshape(4, 4, 3)
    lr = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    hr_aug, lr_aug = PairedAugmenter.apply_with_params(hr, lr, do_h=True, do_v=True, rot_k=1)
    assert hr_aug.shape == hr.shape
    assert lr_aug.shape == lr.shape

"""Dataset loading and occlusion noise."""

import os

import numpy as np
from PIL import Image

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# name -> (folder, downscale factor). Reduced image sizes are 30x37 for ORL
# and 42x48 for Extended YaleB, as suggested in the assignment notebook.
DATASETS = {
    'ORL': ('ORL', 3),
    'YaleB': ('CroppedYaleB', 4),
}


def load_dataset(name, root=DATA_ROOT):
    """Load a dataset as (V, labels, (height, width)).

    V has one vectorised image per column, scaled to [0, 1].
    """
    folder, reduce = DATASETS[name]
    path = os.path.join(root, folder)
    images, labels, shape = [], [], None
    # Number only the subject folders, so stray files (README, .DS_Store)
    # cannot shift the labels.
    people = sorted(p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p)))
    for i, person in enumerate(people):
        person_dir = os.path.join(path, person)
        for fname in sorted(os.listdir(person_dir)):
            # Ambient images in Extended YaleB are background-only shots.
            if not fname.endswith('.pgm') or fname.endswith('Ambient.pgm'):
                continue
            img = Image.open(os.path.join(person_dir, fname)).convert('L')
            img = img.resize([s // reduce for s in img.size])
            arr = np.asarray(img, dtype=np.float32) / 255.0
            shape = arr.shape
            images.append(arr.reshape(-1))
            labels.append(i)
    return np.stack(images, axis=1), np.array(labels), shape


def add_block_occlusion(V, shape, block_size, n_blocks, rng):
    """Paste `n_blocks` white squares of side `block_size` into every image.

    Each block is placed independently at a uniformly random position that
    keeps it fully inside the image (blocks may overlap). Pixels are set to
    1.0, i.e. the value 255 on the original grey scale.

    Returns the corrupted matrix and the boolean mask of occluded pixels.
    """
    h, w = shape
    n = V.shape[1]
    mask = np.zeros((h, w, n), dtype=bool)
    for j in range(n):
        for _ in range(n_blocks):
            r = rng.integers(0, h - block_size + 1)
            c = rng.integers(0, w - block_size + 1)
            mask[r:r + block_size, c:c + block_size, j] = True
    mask = mask.reshape(h * w, n)
    V_noisy = V.copy()
    V_noisy[mask] = 1.0
    return V_noisy, mask

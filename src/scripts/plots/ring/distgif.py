import os
import argparse
from typing import Callable

import matplotlib.cm as cm
import numpy as np
from PIL import Image as pillow
import cv2

parser = argparse.ArgumentParser(
    description="Create GIF from distributions"
)
parser.add_argument('path', type=str, default="checkpoints/gaussian-ring")
parser.add_argument('--max-num-frames', type=int, default=150, help="The maximum number of frames")
parser.add_argument('--copy-frames', type=int, default=30, help="The number of frames to copy at the end")
parser.add_argument('--gif-size', type=int, default=256, help="The width and height of the gifs")
parser.add_argument('--duration', type=int, default=100, help="The duration of each frame in ms")
parser.add_argument('--drop-last-frames', type=int, default=0, help="The number of last frames to drop")

"""
python -m scripts.plots.ring.distgif checkpoints/gaussian-ring --drop-last-frames 224
"""


if __name__ == '__main__':
    def to_rgb(x: np.ndarray, cmap: cm.ScalarMappable, cmap_transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        #x = x[51:-50, 51:-50]
        x = (cmap.to_rgba(cmap_transform(x.T)) * 255.0).astype(np.uint8)[..., :-1]
        if x.shape[0] != args.gif_size or x.shape[1] != args.gif_size:
            x = cv2.resize(x, dsize=(args.gif_size, args.gif_size), interpolation=cv2.INTER_CUBIC)
        return x

    def to_rgb_image(x: np.ndarray) -> pillow.Image:
        return pillow.fromarray(x, mode='RGB')

    print("Loading the GIF data ...")

    args = parser.parse_args()
    checkpoint_paths = [
        f"{args.path}/ring/MonotonicPC/RGran_R1_K2_D1_Lcp_OAdam_LR0.005_BS64_IU",
        f"{args.path}/ring/MonotonicPC/RGran_R1_K16_D1_Lcp_OAdam_LR0.005_BS64_IU",
        f"{args.path}/ring/BornPC/RGran_R1_K2_D1_Lcp_OAdam_LR0.001_BS64_IN"
    ]
    gt_array = np.load(os.path.join(checkpoint_paths[0], 'gt.npy'))
    gt_array = np.broadcast_to(gt_array, (args.max_num_frames, gt_array.shape[0], gt_array.shape[1]))
    arrays = map(lambda p: np.load(os.path.join(p, 'diststeps.npy')), checkpoint_paths)
    if args.drop_last_frames > 0:
        arrays = map(lambda a: a[:-args.drop_last_frames], arrays)
    arrays = [gt_array] + list(arrays)

    print("Constructing the GIF ...")

    num_frames = min(args.max_num_frames, min(len(a) for a in arrays))
    frames_idx = [np.linspace(0.0, 1.0, num=num_frames + 1, endpoint=True)[:-1] for _ in range(len(arrays))]
    arrays_idx = list(map(lambda x: np.floor(x[0] * len(x[1])).astype(np.int64), zip(frames_idx, arrays)))
    arrays = list(map(lambda x: x[1][x[0]], zip(arrays_idx, arrays)))

    cmap_transform = lambda x: np.tanh(2.0 + 0.8 * np.log(x))
    cmap_min = cmap_transform(min(np.min(a) for a in arrays))
    cmap_max = cmap_transform(max(np.max(a) for a in arrays))
    cmap = cm.ScalarMappable(cm.colors.Normalize(cmap_min, cmap_max), cmap='turbo')
    arrays = map(
        lambda a: np.array([to_rgb(a[i], cmap, cmap_transform) for i in range(num_frames)]),
        arrays
    )

    caption_height = 48
    font, fontscale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    arrays = map(
        lambda x: np.concatenate([
            x[1], np.tile(cv2.putText(
                np.full(fill_value=255, shape=(caption_height, x[1].shape[2], 3), dtype=np.uint8),
                x[0],
                (int(0.5 * (x[1].shape[2] - cv2.getTextSize(x[0], font, fontscale, thickness)[0][0])),
                    int(0.5 * (caption_height + cv2.getTextSize(x[0], font, fontscale, thickness)[0][1]))),
                font, fontscale, (16, 16, 16), thickness, cv2.LINE_AA), reps=(num_frames, 1, 1, 1))
        ], axis=1), zip(['Ground Truth'] + ['GMM (K=2)', 'GMM (K=16)', 'Squared SGMM'], arrays)
    )
    gif_images = np.concatenate(list(arrays), axis=2)

    print("Saving GIF to file ...")

    gif_iterator = (
        (to_rgb_image(gif_images[i]) if i < len(gif_images) else to_rgb_image(gif_images[-1]))
        for i in range(len(gif_images) + args.copy_frames)
    )
    img = next(gif_iterator)
    with open(os.path.join('figures', 'gaussian-ring', f'learning-distributions.gif'), 'wb') as fp:
        img.save(fp=fp, format='GIF', append_images=gif_iterator, save_all=True, duration=args.duration, loop=0, optimize=True)

"""LGI
Local group invariance for heart rate estimation from face videos.
Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
(2018).
"""

import math
import numpy as np
from scipy import linalg
from scipy import signal


def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)


def LGI(frames):
    precessed_data = process_video(frames)
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp
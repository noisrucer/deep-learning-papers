import numpy as np
import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 base_sizes=None,
                 ratios=None,
                 scales=None
                 ):
        super().__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if base_sizes is None:
            self.base_sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])


    def forward(self, img):
        img_shape = np.array(img.shape[2:]) # (W, H)

        # Projected size of each pixel in feature map
        # W + 2 ** x - 1 // 2 ** x to cover the last index
        img_shapes = [(img_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.base_sizes[idx],
                                       ratios=self.ratios,
                                       scales=self.scales)
            shifted_anchors = shift(img_shapes[idx], self.strides[idx], anchors) # (28*9, 4)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))



def generate_anchors(base_size=16, ratios=None, scales=None):
    n_anchors = len(ratios) * len(scales)

    anchors = np.zeros((n_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors # (9, 4)


def shift(shape, stride, anchors):
    # anchors: (9, 4)
    shift_x = (np.arange(1, shape[1]) + 0.5) * stride # middle point
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride # middle point

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose() # shifts = (28, 4)


    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0] # 9
    K = shifts.shape[0] # 28
    all_anchors = (anchors.reshape((1, A, 4)) # (1, 9, 4)
                   + shifts.reshape((1, K, 4)).transpose((1, 0, 2)) # (28, 1, 4)
                   )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


# ######################
# This is our trained model based on the CARVE dataset
# ######################

from foundation_model.model import *
from analysis.filter.filter_3D import frangi_filter_scan, frangi_filter
from analysis.filter.frangi_gpu import frangi_filter_gpu
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from visualization.view_3D import *
import numpy as np
from HiPaS.release.models import *


def predict_intra_av_2(scan):
    model = MedNext(in_channels=2,
                      n_classes=2,
                      n_channels=24,
                      kernel_size=3,
                      exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                      do_res=True,
                      block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                      deep_supervision=True)

    pretrained_model = torch.load("./CARVE.pth")
    model.load_state_dict(pretrained_model, strict=True)

    model.half()
    model.eval()
    model = model.to('cuda')
    input_ct = torch.tensor(scan[np.newaxis]).to(torch.float).half()
    with torch.no_grad():
        pre = sliding_window_inference(inputs=input_ct,
                                       predictor=model,
                                       roi_size=(192, 192, 160),
                                       sw_batch_size=2,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.125,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")
    pre = pre[0].detach().cpu().numpy()[0]
    pre = np.array(pre > 0.51, "float32")
    return pre

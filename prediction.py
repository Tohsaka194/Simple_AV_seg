from foundation_model.model import *
from analysis.filter.filter_3D import frangi_filter_scan, frangi_filter
from analysis.filter.frangi_gpu import frangi_filter_gpu
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from visualization.view_3D import *
import numpy as np
from .models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


def predict_zoomed(ct_array):
    ct_array = np.clip(ct_array, 0, 1)
    ct_array = torch.tensor(ct_array[np.newaxis, np.newaxis]).to(torch.float).half()
    zoomed_ct = F.interpolate(ct_array, scale_factor=0.5, mode="trilinear", align_corners=False)

    ###########################################################
    model_lung = UNet(in_channel=1, num_classes=2, active="softmax")
    model_lung.load_state_dict(torch.load("./lung.pth"))
    model_lung.half()
    model_lung.eval()
    model_lung = model_lung.cuda()

    lung_ct = torch.clamp((zoomed_ct * 1600 - 1000 + 600) / 800, 0, 1)

    with torch.no_grad():
        zoom_lung = sliding_window_inference(inputs=lung_ct,
                                             predictor=model_lung,
                                             roi_size=(256, 256, 256),
                                             sw_batch_size=1,
                                             overlap=0.5,
                                             mode="gaussian",
                                             sigma_scale=0.125,
                                             progress=False,
                                             sw_device="cuda",
                                             device="cpu")

    pre_lung = F.interpolate(zoom_lung, scale_factor=2, mode="trilinear", align_corners=False)
    pre_lung = pre_lung.detach().cpu().numpy()[0, 0]
    pre_lung = np.array(pre_lung > 0.51, "float32")

    # return pre_lung
    ##########################################################
    model_av = UNet(in_channel=1, num_classes=3, active="softmax")
    model_av.load_state_dict(torch.load("./main_AV.pth"))
    model_av.half()
    model_av.eval()
    model_av.cuda()

    with torch.no_grad():
        pre_av = sliding_window_inference(inputs=zoomed_ct,
                                          predictor=model_av,
                                          roi_size=(256, 256, 256),
                                          sw_batch_size=1,
                                          overlap=0.25,
                                          mode="gaussian",
                                          sigma_scale=0.125,
                                          progress=False,
                                          sw_device="cuda",
                                          device="cpu")
    pre_av = F.interpolate(pre_av, scale_factor=2, mode="trilinear", align_corners=False)
    pre_av = pre_av.detach().cpu().numpy()[0]

    pre_artery = np.array(pre_av[0] > 0.52, "float32")
    pre_vein = np.array(pre_av[1] > 0.52, "float32")

    return pre_artery, pre_vein, pre_lung


def predict_intra_av_1(scan):
    model_1 = MedNext(in_channels=3,
                      n_classes=2,
                      n_channels=24,
                      kernel_size=3,
                      exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                      do_res=True,
                      block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                      deep_supervision=True)

    pretrained_model = torch.load("./AV_stage_1.pth")
    model_1.load_state_dict(pretrained_model, strict=True)

    model_1.half()
    model_1.eval()
    model_1 = model_1.to('cuda')
    input_ct = torch.tensor(scan[np.newaxis]).to(torch.float).half()

    with torch.no_grad():
        pre = sliding_window_inference(inputs=input_ct,
                                       predictor=model_1,
                                       roi_size=(192, 192, 160),
                                       sw_batch_size=2,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.125,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")
    pre = pre[0].detach().cpu().numpy()[0]
    return pre


def predict_whole_av(ct_array, artery_p=None, vein_p=None, lung=None):
    if ct_array.shape[-1] % 2 == 1:
        ct_array = ct_array[:, :, :-1]

    if (artery_p is None) or (vein_p is None) or (lung is None):
        artery_p, vein_p, lung = predict_zoomed(ct_array)

    new_array = np.clip(ct_array, 0, 1)
    loc = np.array(np.where(lung > 0))
    x_min, x_max = np.min(loc[0]), np.max(loc[0])
    y_min, y_max = np.min(loc[1]), np.max(loc[1])
    z_min, z_max = np.min(loc[2]), np.max(loc[2])

    filtered = frangi_filter_gpu(new_array[x_min:x_max, y_min:y_max, z_min:z_max], sigma=[0.5, 1, 1.5],
                                 transfer_device=True, device="cuda")
    prior_1 = np.array(artery_p * 0.25 + vein_p * 0.75)

    input_set_1 = np.concatenate((new_array[x_min:x_max, y_min:y_max, z_min:z_max][np.newaxis],
                                  prior_1[x_min:x_max, y_min:y_max, z_min:z_max][np.newaxis],
                                  filtered[np.newaxis]),
                                 axis=0)
    pre_1 = predict_intra_av_1(input_set_1)
    pred_1 = np.zeros([2, ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]])
    pred_1[0, x_min:x_max, y_min:y_max, z_min:z_max] = pre_1[0]
    pred_1[1, x_min:x_max, y_min:y_max, z_min:z_max] = pre_1[1]

    a = np.array(pred_1[0] > 0.5, "float32")
    v = np.array(pred_1[1] > 0.5, "float32")
    return a, v, lung


def predict_airway_test(ct_array, lung=None):
    model = MedNeXt_seg(in_channels=2,
                        n_channels=24,
                        n_classes=1,
                        exp_r=[2, 4, 8, 16, 16],
                        kernel_size=3,
                        do_res=True,
                        do_res_up_down=True,
                        block_counts=[2, 4, 8, 16, 32])

    model.load_state_dict(torch.load("./airway.pth"))
    model = model.to(torch.float).half()
    model = model.cuda()

    if lung is not None:
        loc = np.array(np.where(lung > 0))
        x_min, x_max = np.min(loc[0]), np.max(loc[0])
        y_min, y_max = np.min(loc[1]), np.max(loc[1])
        z_min, z_max = np.min(loc[2]), np.max(loc[2])

        ct_array = ct_array[x_min:x_max, y_min:y_max, z_min:z_max]

    filtered = frangi_filter(ct_array)
    # filtered = frangi_filter_gpu(ct_array, transfer_device=True, transpose=True, device=device)
    input_ct = torch.tensor(np.stack((ct_array, filtered), axis=0)[np.newaxis]).to(torch.float).half()

    with torch.no_grad():
        pre = sliding_window_inference(inputs=input_ct,
                                       predictor=model,
                                       roi_size=(192, 192, 160),
                                       sw_batch_size=2,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.125,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")

    pre = torch.sigmoid(pre).detach().cpu().numpy()[0, 0]
    # print(pre.shape)
    pre = np.array(pre > 0.51, "float32")

    if lung is not None:
        airway = np.zeros([512, 512, ct_array.shape[-1]])
        airway[x_min:x_max, y_min:y_max, z_min:z_max] = pre
    else:
        airway = pre
    return airway


def predict_red_img(ct_slice, normalization=True):
    if not normalization:
        ct_slice = ct_slice * 1600 - 1000

    ct_slice = np.clip((ct_slice * 1600) / (3000 + 1000), 0, 1)

    ct_scan = torch.tensor(ct_slice[np.newaxis, np.newaxis, :]).to(torch.float).cuda()
    denoise_model = RED_CNN()
    denoise_model.load_state_dict(torch.load("RED_CNN.ckpt"))
    denoise_model = denoise_model.cuda()
    denoise_model = denoise_model.to('cuda')
    prediction = denoise_model(ct_scan).cpu().detach().numpy()[0, 0]

    if normalization:
        return np.clip((prediction * 4000) / 1600, 0, 1)
    else:
        return prediction * 4000 - 1000


def predict_red_scan(ct_scan):
    denoised_result = np.zeros(ct_scan.shape)
    for i in range(ct_scan.shape[-1]):
        denoised_result[:, :, i] = predict_red_img(ct_scan[:, :, i])
    return denoised_result


if __name__ == '__main__':
    ct_array = np.load("/data_backup/chest_CT/ct_array/HiPaS_CTPA/ct_scan/AL00004.npz", allow_pickle=True)["data"]
    ct_array = np.clip((ct_array + 1000.0) / 1600, 0, 1)
    ct_array = predict_red_scan(ct_array)  # denoising can help better segmentation

    a, v, lung = predict_whole_av(ct_array)

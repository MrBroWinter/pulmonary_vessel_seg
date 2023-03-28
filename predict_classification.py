import time
import os
import torch
from cfg import cfg
import model_zoo
import numpy as np
import SimpleITK as sitk
import threading
from torch.nn import functional as F
from multiprocessing import cpu_count, Pool
from data_preprocess import PreProcess
import itertools
from functools import partial
import math
import os

def overlap_predict(net, ct, patch_size, classes, rate=2):
    """
    :param net: 网络模型
    :param data_for_predict:   输入的待预测矩阵
    :param patch_size:     切块大小
    :param rate=2   1/2重叠滑块
    :return: 预测输出结果   与输入大小一致
    """

    # slide_rate = patch_size // rate  # 滑块的步长，默认为patch_size的一半（64）
    slide_rate = list(map(lambda x: x // rate, patch_size))
    pad_num_z = (patch_size[0] - ct.shape[0] % patch_size[0]) if (
            ct.shape[0] <= patch_size[0]) else (slide_rate[0] - ct.shape[0] % slide_rate[0])
    pad_num_y = (patch_size[1] - ct.shape[1] % patch_size[1]) if (
            ct.shape[1] <= patch_size[1]) else (slide_rate[1] - ct.shape[1] % slide_rate[1])
    pad_num_x = (patch_size[2] - ct.shape[2] % patch_size[2]) if (
            ct.shape[2] <= patch_size[2]) else (slide_rate[2] - ct.shape[2] % slide_rate[2])

    tmp_ct = np.pad(ct, ((0, pad_num_z), (0, pad_num_y), (0, pad_num_x)), 'constant')


    z_slide_num = math.ceil((tmp_ct.shape[0] - patch_size[0]) / slide_rate[0]) + 1
    y_slide_num = math.ceil((tmp_ct.shape[1] - patch_size[1]) / slide_rate[1]) + 1
    x_slide_num = math.ceil((tmp_ct.shape[2] - patch_size[2]) / slide_rate[2]) + 1
    # 保存最终的预测结果
    tmp_res = np.repeat(np.zeros(tmp_ct.shape)[None, ...], classes, axis=0)

    with torch.no_grad():
        for xx in range(x_slide_num):
            for yy in range(y_slide_num):
                for zz in range(z_slide_num):
                    ct_part = tmp_ct[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                              yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                              xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]]

                    ct_tensor = torch.FloatTensor(ct_part).cuda()
                    ct_tensor = ct_tensor.unsqueeze(dim=0)
                    ct_tensor = ct_tensor.unsqueeze(dim=0)

                    # print(ct_tensor.shape)
                    outputs = net(ct_tensor)
                    outputs = outputs.squeeze().cpu().detach().numpy()
                    # 将预测的结果加入到对应的位置上
                    tmp_res[:, zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += outputs

    return tmp_res[:, 0:ct.shape[0], 0:ct.shape[1], 0:ct.shape[2]]


def get_path(test_root_path):
    result = []
    for sub in os.listdir(test_root_path):
        result.append(os.path.join(test_root_path, sub))
    return result

def predict(cfg, net, patient_path):
    p = PreProcess(cfg)
    processed_ct, origial_size = p.read_and_normalization_CT(patient_path)
    patient_dir = os.path.dirname(patient_path)

    with torch.no_grad():
        res_logits = overlap_predict(net, processed_ct, cfg.patch_size, cfg.classes, rate=2)  # 预测结果
        res = np.argmax(res_logits, axis=0)
    res_itk = sitk.GetImageFromArray(res)
    sitk.WriteImage(res_itk, os.path.join(patient_dir, "predict.nii.gz"))




if __name__ == '__main__':
    # thead pool
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_time = time.time()
    patient_path_list = get_path(cfg.test_root_path)

    model, _ = model_zoo.create_model(cfg)


    model.load_state_dict(torch.load(cfg.used_ckpt)['model_state_dict'])
    model.to(cfg.device)
    model.eval()

    res_list = []

    for sub in patient_path_list:
        sub_result = predict(cfg, model, sub)
        print(sub_result)
        # patient_name = sub.split("\\")[-1]
        # res_list.append(sub_result[patient_name])

    print("time:{}".format(time.time() - start_time))
    print("mean dice:", sum(res_list) / len(res_list))




import os
from easydict import EasyDict
import torch
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


cfg = EasyDict()
# for data process
cfg.patch_size = (96, 128, 128)
cfg.median_spacing = (0.74, 0.74, 1)
cfg.data_path = r"/Files/Xiaozhuang.Wang/PulmonaryData_100_425"
cfg.save_root_path_for_train = r"C:\dongdong.zhao\project\Lung_vessel\train_data"                       # 训练数据切块后保存路径
cfg.save_root_path_for_val = r"C:\dongdong.zhao\project\Lung_vessel\train_data"                     # 验证数据切块后保存路径
cfg.item = 'Vessels'

if cfg.item == 'Vessels':
    cfg.upper = 482
    cfg.lower = -747
    cfg.mean = 123.05
    cfg.std = 230.95



# for train
cfg.inChannels = 1
cfg.classes = 3
cfg.batch_size = 1
cfg.lr = 1e-3
cfg.prior_mask = False
cfg.start_epoch = 0
cfg.train_epochs = 100
cfg.save_frequency = 2
cfg.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cup')
cfg.resume = False
cfg.module_path = r"./checkpoint/Epoch_52_loss_0.225.pth"
cfg.module_save_root_path = r"./checkpoint"

cfg.train_data_path = os.path.join(cfg.save_root_path_for_train, "ct_patch")
cfg.val_data_path = os.path.join(cfg.save_root_path_for_val, "ct_patch")


# model list:   'UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET',
#               "SKIPDENSENET3D","DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D",
#               "COVIDNET1", "COVIDNET2", "CNN", "HIGHRESNET", "WNET", "UNET3D_ours", "WingsNet", "ResUnetPlusPlus"
cfg.model = 'ResUnetPlusPlus'
cfg.opt = 'SGD'    #optional: SGD, ADAM, RMSPROP

# for predict
cfg.threshold = 0.5
cfg.test_root_path = r"C:\dongdong.zhao\project\Lung_vessel\val_raw_data"
cfg.test_save_path = r"C:\dongdong.zhao\project\Lung_vessel\val_raw_data_predict"
cfg.used_ckpt = r"./checkpoint/Epoch_256_loss_0.1324_unetpp_classfication.pth"

if __name__ == '__main__':
    print(cfg.mean)

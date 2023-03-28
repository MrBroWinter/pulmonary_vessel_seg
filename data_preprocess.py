import os
import SimpleITK as sitk
from utils.utils import read_dicom, Resampling, get_boundaries_from_mask
import numpy as np
import math
import random
import time
from cfg import cfg
class PreProcess(object):
    def __init__(self, cfg):
        self.upper = cfg.upper
        self.lower = cfg.lower
        self.mean = cfg.mean
        self.std = cfg.std
        self.data_path = cfg.data_path
        self.save_root_path_for_train = cfg.save_root_path_for_train
        self.save_root_path_for_val = cfg.save_root_path_for_val
        self.item = cfg.item
        self.median_spacing = cfg.median_spacing
        self.patch_size = tuple(cfg.patch_size)


    def read_and_normalization_CT(self, patient_ct_path):
        ct_itk = sitk.ReadImage(os.path.join(patient_ct_path, "dicom.nii.gz"))
        # ct_itk = read_dicom(patient_ct_path)
        size = ct_itk.GetSize()
        original_size = (size[2], size[1], size[0])
        ct_resampled = Resampling(ct_itk, self.median_spacing, label=False)
        ct_array = sitk.GetArrayFromImage(ct_resampled)

        # ct normalization
        ct_array = ct_array.astype(np.float)
        ct_array = np.clip(ct_array, self.lower, self.upper)
        ct_array = (ct_array - self.mean) / self.std

        return ct_array, original_size

    def read_and_process_label(self, patient_label_path):
        label_itk = sitk.ReadImage(patient_label_path)
        label_resampled = Resampling(label_itk, self.median_spacing, label=True)
        label_array = sitk.GetArrayFromImage(label_resampled)
        # label_array[label_array  > 0] = 1
        return label_array

    def crop_and_save_patch(self, ct_array, label_array, patient_name, save_root_path):
        
        # crop the training patch in a way of slide window
        def slide_crop():
            z_times = math.ceil((bbox['zmax'] - bbox['zmin']) / half_patch_size[0])
            y_times = math.ceil((bbox['ymax'] - bbox['ymin']) / half_patch_size[1])
            x_times = math.ceil((bbox['xmax'] - bbox['xmin']) / half_patch_size[2])

            for i in range(z_times):
                for j in range(y_times):
                    for k in range(x_times):
                        cropped_ct = ct_array[(bbox['zmin'] + (i - 1) * half_patch_size[0]):(bbox['zmin'] + (i + 1) * half_patch_size[0]), 
                                              (bbox['ymin'] + (j - 1) * half_patch_size[1]):(bbox['ymin'] + (j + 1) * half_patch_size[1]), 
                                              (bbox['xmin'] + (k - 1) * half_patch_size[2]):(bbox['xmin'] + (k + 1) * half_patch_size[2])]
                        cropped_label = label_array[(bbox['zmin'] + (i - 1) * half_patch_size[0]): (bbox['zmin'] + (i + 1) * half_patch_size[0]),
                                                    (bbox['ymin'] + (j - 1) * half_patch_size[1]): (bbox['ymin'] + (j + 1) * half_patch_size[1]),
                                                    (bbox['xmin'] + (k - 1) * half_patch_size[2]): (bbox['xmin'] + (k + 1) * half_patch_size[2])]
                        if cropped_ct.shape == self.patch_size and cropped_label.shape == self.patch_size and cropped_label.max() > 0:
                            save_name = "slide_" + str(i) + "_" + str(j) + "_" + str(k) + ".nii.gz"
                            slide_cropped_ct_itk = sitk.GetImageFromArray(cropped_ct)
                            slide_cropped_label_itk = sitk.GetImageFromArray(cropped_label)

                            sitk.WriteImage(slide_cropped_ct_itk, os.path.join(save_root_path, "ct_patch", patient_name, save_name))
                            sitk.WriteImage(slide_cropped_label_itk, os.path.join(save_root_path, "label_patch", patient_name, save_name))

        def random_patch(num_random_patches=20):
            zmin = bbox['zmin']
            zmax = bbox['zmax']
            ymin = bbox['ymin']
            ymax = bbox['ymax']
            xmin = bbox['xmin']
            xmax = bbox['xmax']

            for random_index in range(num_random_patches):
                z_center = np.clip(random.randint(zmin, zmax), half_patch_size[0], ct_array.shape[0] - half_patch_size[0])
                y_center = np.clip(random.randint(ymin, ymax), half_patch_size[1], ct_array.shape[1] - half_patch_size[1])
                x_center = np.clip(random.randint(xmin, xmax), half_patch_size[2], ct_array.shape[2] - half_patch_size[2])

                random_ct_patch = ct_array[(z_center - half_patch_size[0]):(z_center + half_patch_size[0]),
                                           (y_center - half_patch_size[1]):(y_center + half_patch_size[1]),
                                           (x_center - half_patch_size[2]):(x_center + half_patch_size[2])]

                random_label_patch = label_array[(z_center - half_patch_size[0]):(z_center + half_patch_size[0]),
                                                 (y_center - half_patch_size[1]):(y_center + half_patch_size[1]),
                                                 (x_center - half_patch_size[2]):(x_center + half_patch_size[2])]




                if random_ct_patch.shape == self.patch_size and random_label_patch.shape == self.patch_size and random_label_patch.max() > 0:
                    save_name = "random_" + str(random_index) + ".nii.gz"
                    random_cropped_ct_itk = sitk.GetImageFromArray(random_ct_patch)
                    random_cropped_label_itk = sitk.GetImageFromArray(random_label_patch)
                    sitk.WriteImage(random_cropped_ct_itk, os.path.join(save_root_path, "ct_patch", patient_name, save_name))
                    sitk.WriteImage(random_cropped_label_itk, os.path.join(save_root_path, "label_patch", patient_name, save_name))




        if not os.path.exists(os.path.join(save_root_path, "ct_patch", patient_name)):
            os.makedirs(os.path.join(save_root_path, "ct_patch", patient_name))
        if not os.path.exists(os.path.join(save_root_path, "label_patch", patient_name)):
            os.makedirs(os.path.join(save_root_path, "label_patch", patient_name))

        ROI = label_array.copy()
        bbox = get_boundaries_from_mask(ROI)
        half_patch_size = list(map(lambda x: x // 2, self.patch_size))

        # slide_crop()
        random_patch(20)
    
    def train_val_split(self, txt_path, save_path):
        f = open(txt_path, 'r')
        patient_list = f.readlines()
        for sub in patient_list:
            try:
                sub = sub.strip('\n')
                ct_path = os.path.join(self.data_path, sub, "dicom.nii.gz")
                PulmonaryArtery_path = os.path.join(self.data_path, sub, "PulmonaryArtery.nii.gz")
                PulmonaryVein_path = os.path.join(self.data_path, sub, "PulmonaryVeins.nii.gz")
                 
                ct_array, _ = self.read_and_normalization_CT(ct_path)
            
                PulmonaryArtery_array = self.read_and_process_label(PulmonaryArtery_path)
                
                PulmonaryVein_array = self.read_and_process_label(PulmonaryVein_path)
                # print("*3")
                PulmonaryVessels_array = PulmonaryArtery_array + PulmonaryVein_array * 2
                # print("*4")
                self.crop_and_save_patch(ct_array, PulmonaryVessels_array, sub, save_path)
            except:
                # print(sub, "is incomplete")
                pass
    def run(self):
        self.train_val_split("/Files/Dongdong.Zhao/code/lung_vessel/data/train_Vessels_new.txt", self.save_root_path_for_train)
        self.train_val_split("/Files/Dongdong.Zhao/code/lung_vessel/data/valid_Vessels_new.txt", self.save_root_path_for_val)
if __name__ == '__main__':
    p = PreProcess(cfg)
    p.run()
 







    











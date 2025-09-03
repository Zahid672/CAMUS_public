import os
import numpy as np
import cv2
import SimpleITK as sitk
import random
from pathlib import Path
import h5py
import collections
import tqdm as tqdm

# import echonet
# import echonet_utils

import pandas
import skimage.draw

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def prepare_cfg(path): # This function reads a configuration file and returns a dictionary with key-value pairs.
    f = open(path, 'r')
    info = f.read().splitlines()

    data_dict = {}
    for item in info:
        key, value = item.split(': ')
        data_dict[key] = value
    
    return data_dict

class CAMUS_loader(Dataset):
    def __init__(self, data_path, patient_list_path, view):
        super().__init__()
        self.data_path = data_path
        self.patient_list_path = patient_list_path
        self.patient_list_path = np.load(self.patient_list_path)

        assert view in ['2CH', '4CH']
        self.view = view
        self.instants = ['ED', 'ES']

        self.data_transform = {
            "image": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(degrees = (0, 180)),
                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "gt": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation = InterpolationMode.NEAREST),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        }

    def __len__(self) -> int:
        return len(self.patient_list_path)
    
    def __getitem__(self, index):
        rnum = random.randint(0, 1)

        image_pattern = "{patient_name}_{view}_{instant}.nii.gz"
        gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"

        patient_name, instant = self.patient_list_path[index]
        patient_dir = os.path.join(self.data_path, patient_name)

        image, info_image = self.sitk_load(os.path.join(patient_dir, image_pattern.format(patient_name = patient_name, view = self.view, instant = instant)))
        gt, info_gt = self.sitk_load(os.path.join(patient_dir, gt_mask_pattern.format(patient_name = patient_name, view = self.view, instant = instant)))

        cropped_gt = np.where(image == 0, 0, gt)

        # print('info image: ', info_image)
        # print('info gt: ', info_gt)
        # image = cv2.resize(image, (224, 224))
        # gt = cv2.resize(gt, (224, 224), interpolation = cv2.INTER_NEAREST)
        print('original shapes: ', image.shape, gt.shape)
        print('image range: ', np.min(image), np.max(image))
        image = self.data_transform['image'](image)
        # gt = self.data_transform['gt'](cropped_gt)
        gt    = self.data_transform['gt'](gt).squeeze(0).long() # [H,W]  <- remove the 1 channel
        

        return image, gt
    
    def sitk_load(self, filepath: str):
        """Loads an image using SimpleITK and returns the image and its metadata.

        Args:
            filepath: Path to the image.

        Returns:
            - ([N], H, W), Image array.
            - Collection of metadata.
        """
        # Load image and save info
        image = sitk.ReadImage(str(filepath))
        info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

        # Extract numpy array from the SimpleITK image object
        im_array = np.squeeze(sitk.GetArrayFromImage(image))

        return im_array, info
    

class Prepare_CAMUS(): ## This class is used to prepare the CAMUS dataset for training and testing.
    """
    This class prepares the CAMUS dataset by separating patients into high, medium, and low quality based on the 'ImageQuality' field in the Info_2CH.cfg file.
    It also separates the dataset into training and testing sets.
    """
    
    def __init__(self, data_path, save_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.save_path = save_path

    def prepare(self): ## This function checks if the necessary files already exist. If they do, it prints a message and does not perform any further actions. 
        #If the files do not exist, it calls the methods to separate patients and create training/testing sets.
        if Path(os.path.join(self.save_path, 'train_samples.npy')).is_file() and Path(os.path.join(self.save_path, 'test_ED.npy')).is_file():
            print('\n', '----- The files already exist -----')
        else:
            self.separate_patients(self.data_path)
            self.separate_train_test()

    def create_rep(self, orig_array, mode = 'combined'): # This function creates a representation of the original array by appending 'ED' and 'ES' labels to each sample.
        if mode == 'combined':
            rep_array = []
            for sample in orig_array:
                rep_array.append([sample, 'ED'])
                rep_array.append([sample, 'ES'])
            return rep_array
        
        elif mode == 'separate':
            rep_array_1 = []
            rep_array_2 = []
            for sample in orig_array:
                rep_array_1.append([sample, 'ED'])
                rep_array_2.append([sample, 'ES'])
            return rep_array_1, rep_array_2
        


    def separate_train_test(self): # This function separates the patients into training and testing sets based on their quality.
        np.random.shuffle(self.high_quality_patients)
        np.random.shuffle(self.medium_quality_patients)
        np.random.shuffle(self.low_quality_patients)

        h_len = len(self.high_quality_patients)
        m_len = len(self.medium_quality_patients)
        l_len = len(self.low_quality_patients)
        h_samps = 20#int(h_len*0.2)
        m_samps = 20#int(m_len*0.2)
        l_samps = 10#int(l_len*0.2)

        htest = self.high_quality_patients[0:h_samps]
        htrain = self.high_quality_patients[h_samps:h_len]

        mtest = self.medium_quality_patients[0:m_samps]
        mtrain = self.medium_quality_patients[m_samps:m_len]

        ltest = self.low_quality_patients[0:l_samps]
        ltrain = self.low_quality_patients[l_samps:l_len]

        htest = np.array(htest)
        mtest = np.array(mtest)
        ltest = np.array(ltest)
        
        total_test = np.array(np.concatenate((htest, mtest, ltest)))
        total_training = np.array(np.concatenate((htrain, mtrain, ltrain)))

        total_test_ED, total_test_ES = self.create_rep(total_test, mode = 'separate')
        total_training = self.create_rep(total_training)

        train_save_path = os.path.join(self.save_path, 'train_samples.npy')
        test_ED_path = os.path.join(self.save_path, 'test_ED.npy')
        test_ES_path = os.path.join(self.save_path, 'test_ES.npy')
        np.save(train_save_path, total_training)
        np.save(test_ED_path, total_test_ED)
        np.save(test_ES_path, total_test_ES)


    def separate_patients(self, folder_path): # This function separates patients into high, medium, and low quality based on the 'ImageQuality' field in the Info_2CH.cfg file.
        self.high_quality_patients = []
        self.medium_quality_patients = []
        self.low_quality_patients = []

        for patient in os.listdir(folder_path):
            patient_folder = os.path.join(folder_path, patient)
            # print('patient folder: ', patient_folder)
            if not os.path.isdir(patient_folder):
        # Skip files or non-directories
                continue
            # Now process patient_folder as a directory:
            info_cfg_path = os.path.join(patient_folder, 'Info_2CH.cfg')
            if not os.path.isfile(info_cfg_path):
                print(f"Warning: Config file not found for patient folder {patient_folder}, skipping.")
                continue

            info_dict = prepare_cfg(info_cfg_path)
            quality = info_dict.get('ImageQuality', None)
            if quality == 'Good':
                self.high_quality_patients.append(patient)
            elif quality == 'Medium':
                self.medium_quality_patients.append(patient)
            elif quality == 'Poor':
                self.low_quality_patients.append(patient)
            else:
                print(f"Warning: Unknown ImageQuality '{quality}' in {info_cfg_path}")

        print('High quality patients: ', len(self.high_quality_patients))
        print('Medium quality patients: ', len(self.medium_quality_patients))
        print('Low quality patients: ', len(self.low_quality_patients))




if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join('database_nifti')
    save_dir = os.path.join('prepared_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prepare_obj = Prepare_CAMUS(data_dir, save_dir)
    prepare_obj.prepare()
    # # patient_list_path = os.path.join('patient_list.npy')
    # data_dir = 'E:\Segmentation_Task\CAMUS_public\database_nifti'


    import numpy as np
    train_samples = np.load(os.path.join(save_dir, 'train_samples.npy'), allow_pickle=True)
    test_ED_samples = np.load(os.path.join(save_dir, 'test_ED.npy'), allow_pickle=True)
    test_ES_samples = np.load(os.path.join(save_dir, 'test_ES.npy'), allow_pickle=True)

    print(len(train_samples), len(test_ED_samples), len(test_ES_samples))
    print(train_samples[:2])    





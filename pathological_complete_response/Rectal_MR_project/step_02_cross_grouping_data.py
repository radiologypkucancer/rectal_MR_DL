# ----------------------------------------------------------------------------------
# imports
import os,json
import numpy as np
import shutil

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

# ----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    src_subfolder = 'D:\\data\\sunyingshi\\auged_8_kinds_data_3d_volume_TRG'
    grouped_subfolder = 'D:\\data\\sunyingshi\\cross_grouped_data_for_TRG_label_test'

    raw_img_groups = {'t2_post': 't2_post'}

    other_groups = {'t2_pre': 't2_pre',
                    't2_post': 't2_post',
                     'Kapp_pre': 'Kapp_pre',
                     'Kapp_post': 'Kapp_post',
                     'Dapp_pre': 'Dapp_pre',
                     'Dapp_post': 'Dapp_post',
                    'Blog1000_pre': 'Blog1000_pre',
                    'Blog1000_post': 'Blog1000_post'}

    TRG_label_group = {'0': '0',
                       '1': '1',
                       '2': '2',
                       '3': '3'}

    for TRG in TRG_label_group.keys():
        trg_label = TRG_label_group.get(TRG)
        for group in raw_img_groups.keys():
            lesion_num = 0
            Lesion_config = {}
            Lesion_config['lesions'] = {}
            dcm_root = os.path.join(src_subfolder, raw_img_groups.get(group))
            target_root = os.path.join(grouped_subfolder)
            if not os.path.exists(target_root): os.mkdir(target_root)
            # get all of the lesion ROIs
            for patient_root, subdirs0, files in os.walk(dcm_root):
                if not 'labels.json' in files: continue
                patient_root_vec = patient_root.split(os.path.sep)
                if patient_root_vec[-1] != trg_label: continue
                with open(os.path.join(patient_root, 'labels.json'), 'r', encoding='utf-8') as f_json:
                    label_config0 = json.load(f_json)
                    TRG_label= label_config0['Label']['TRG']
                if  TRG_label != TRG_label_group.get(TRG): continue
                Lesion_config['lesions']['lesion_' + str(lesion_num)] = os.path.join(dcm_root, patient_root_vec[-3],patient_root_vec[-2])
                lesion_num += 1
            # randomize the lesion ROIs
            print('lesion_num:')
            print(lesion_num)
            arr = np.arange(lesion_num)
            np.random.shuffle(arr)
            Lesion_config['randomized lesions']={}
            Lesion_config['randomized lesions']['array'] = {}
            Lesion_config['randomized lesions']['cross_group1'] = {}
            Lesion_config['randomized lesions']['cross_group2'] = {}
            Lesion_config['randomized lesions']['cross_group3'] = {}
            Lesion_config['randomized lesions']['cross_group4'] = {}

            aug_num = 120
            copy_num = 0
            for i in arr:
                Lesion_config['randomized lesions']['array']['array_'+ str(i)] = str(i)
                copy_num += 1
                if copy_num < (lesion_num) * 0.25:
                    my_target_root = os.path.join(target_root, 'cross_group1')
                    Lesion_config['randomized lesions']['cross_group1']['cross_group1_' + str(i)] = str(i)
                elif copy_num < (lesion_num) * 0.5:
                    my_target_root = os.path.join(target_root, 'cross_group2')
                    Lesion_config['randomized lesions']['cross_group2']['cross_group2_' + str(i)] = str(i)
                elif copy_num < (lesion_num) * 0.75:
                    my_target_root = os.path.join(target_root, 'cross_group3')
                    Lesion_config['randomized lesions']['cross_group3']['cross_group3_' + str(i)] = str(i)
                else:
                    my_target_root = os.path.join(target_root, 'cross_group4')
                    Lesion_config['randomized lesions']['cross_group4']['cross_group4_' + str(i)] = str(i)

                src_root =  Lesion_config['lesions']['lesion_' + str(i)]
                with open(os.path.join(src_root, '0', 'labels.json'), 'r', encoding='utf-8') as f_json:
                    label_config0 = json.load(f_json)
                    YPCR_label = label_config0['Label']['YPCR']
                if YPCR_label == '0':
                    aug_num = 30
                else:
                    aug_num = 120
                for group_key in other_groups.keys():
                    group_name = other_groups.get(group_key)
                    my_src_root = src_root.replace('t2_post', group_name)
                    if 'Dapp' in group_name:
                        my_src_root = my_src_root.replace('lesion_patch_1', 'Dapp_lesion_patch_1')
                    elif 'Kapp' in group_name:
                        my_src_root = my_src_root.replace('lesion_patch_1', 'Kapp_lesion_patch_1')
                    elif 'Blog1000' in group_name:
                        my_src_root = my_src_root.replace('lesion_patch_1', 'Blog1000_lesion_patch_1')

                    for aug_idx in range(aug_num):
                        temp_src_root = os.path.join(my_src_root, str(aug_idx))
                        patient_root_vec = temp_src_root.split(os.path.sep)
                        # if not os.path.exists(src_root): continue
                        for patient_root, subdirs1, files in os.walk(temp_src_root):
                            if not 'labels.json' in files: continue
                            for file in files:
                                my_src_file = os.path.join(patient_root, file)
                                my_target_root0 = os.path.join(my_target_root, group_name,
                                                               patient_root_vec[-3],patient_root_vec[-2],patient_root_vec[-1])
                                if not os.path.exists(my_target_root0): os.makedirs(my_target_root0)
                                my_target_file = os.path.join(my_target_root0, file)
                                shutil.copy(my_src_file, my_target_file)

            with open(os.path.join(target_root, 'TRG_' + trg_label + '_lesions.json'), 'w', encoding='utf-8') as f_json:
                json.dump(Lesion_config, f_json, indent=4)


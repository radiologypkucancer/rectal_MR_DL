# ----------------------------------------------------------------------------------
# imports
import numpy as np
import json, os

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()
#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # data root
    src_subfolder = 'D:\\data\\sunyingshi\\cross_grouped_data_for_3kinds_label_Kapp_Dapp_full_lesion_0410'
    grouped_subfolder = os.path.join(src_subfolder, 'model_configs_test', 'model_C_Tdegrade_abyes_no_with_blog1000_data')

    # data phases
    phases = ['t2_pre', 't2_post', 'Kapp_pre', 'Kapp_post', 'Dapp_pre','Dapp_post','Blog1000_pre', 'Blog1000_post']

    # data groups
    train_groups = {'cross_group1':'cross_group1',
                    'cross_group2':'cross_group2'}
    val_groups = ['cross_group4']
    test_groups = ['cross_group3']

    # -----------------------------------------------------------------------------------
    # generate validation config file
    validation_config = {}
    validation_config['validation'] = {}
    lesion_size=0
    i_group = 0
    group_size = 1000
    for val_group in val_groups:
        dcm_root = os.path.join(src_subfolder, val_group, 't2_post')
        for patient_root, subdirs, files in os.walk(dcm_root):
            if not 'labels.json' in files: continue
            patient_root_vec = patient_root.split(os.path.sep)
            aug_num = int(patient_root_vec[-1])
            if aug_num == 0:
                with open(os.path.join(patient_root, 'labels.json'), 'r', encoding='utf-8') as f_json:
                    label_config = json.load(f_json)
                    TRG_Rate = label_config['Label']['downstage']
                    if TRG_Rate =='NULL': continue
                    if int(TRG_Rate) > 0.5:
                        TRG_Rate = 'ab_yes_downstage'
                    elif int(TRG_Rate) < 0.5:
                        TRG_Rate = 'no_downstage'
                    else:continue
                lesion_size += 1
                if i_group % group_size == 0:
                    group_id = 1 + int(i_group / group_size)
                    group_name = 'BJCH_' + str(group_id)
                    validation_config['validation'][group_name] = {}
                sample_name = val_group+ '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                              + '_' +patient_root_vec[-2]+ '_Aug_' + patient_root_vec[-1]
                validation_config['validation'][group_name][sample_name] ={}
                phase_files = []
                image_file = os.path.join(patient_root, 'lesion_patch.nii.gz')
                for phase in phases:
                    phase_name = image_file.replace('t2_post', phase)
                    if 'Dapp' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Dapp_lesion_patch_1')
                    elif 'Kapp' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Kapp_lesion_patch_1')
                    elif 'Blog1000' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Blog1000_lesion_patch_1')
                    phase_files.append(phase_name)
                validation_config['validation'][group_name][sample_name]['images_data_niix'] = phase_files
                validation_config['validation'][group_name][sample_name]['class_description'] = TRG_Rate
                i_group += 1

    target_root = grouped_subfolder
    with open(os.path.join(target_root,  'model_c_val_config.json'), 'w', encoding='utf-8') as f_json:
            json.dump(validation_config, f_json, indent=4)

    # -----------------------------------------------------------------------------------
    # generate training config file
    train_config = {}
    train_config['train'] = {}
    i_group = 0
    group_size = 1000
    lesion_size = -1
    for group in train_groups.keys():
        dcm_root = os.path.join(src_subfolder, train_groups.get(group), 't2_post')
        for patient_root, subdirs, files in os.walk(dcm_root):
            if not 'labels.json' in files: continue
            with open(os.path.join(patient_root, 'labels.json'), 'r', encoding='utf-8') as f_json:
                label_config = json.load(f_json)
                TRG_Rate = label_config['Label']['downstage']
                if TRG_Rate == 'NULL': continue
                if int(TRG_Rate) > 0.5:
                    TRG_Rate = 'ab_yes_downstage'
                elif int(TRG_Rate) < 0.5:
                    TRG_Rate = 'no_downstage'
                else:
                    continue
            patient_root_vec = patient_root.split(os.path.sep)
            lesion_size += 1
            train_config['train']['lesion_' + str(lesion_size)] = {}
            sample_name = train_groups.get(group) + '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                          + '_' + patient_root_vec[-2] + '_Aug_' + patient_root_vec[-1]
            train_config['train']['lesion_' + str(lesion_size)]['sample_lesion_name'] = sample_name
            train_config['train']['lesion_' + str(lesion_size)][sample_name] = {}

            phase_files = []
            image_file = os.path.join(patient_root, 'lesion_patch.nii.gz')
            for phase in phases:
                phase_name = image_file.replace('t2_post', phase)
                if 'Dapp' in phase:
                    phase_name = phase_name.replace('lesion_patch_1', 'Dapp_lesion_patch_1')
                elif 'Kapp' in phase:
                    phase_name = phase_name.replace('lesion_patch_1', 'Kapp_lesion_patch_1')
                elif 'Blog1000' in phase:
                    phase_name = phase_name.replace('lesion_patch_1', 'Blog1000_lesion_patch_1')
                phase_files.append(phase_name)
            train_config['train']['lesion_' + str(lesion_size)][sample_name]['images_data_niix'] = phase_files
            train_config['train']['lesion_' + str(lesion_size)][sample_name]['class_description'] = TRG_Rate
    # randomize the lesion ROIs
    print('training lesion_size:')
    print(lesion_size+1)
    arr = np.arange(lesion_size+1)
    np.random.shuffle(arr)

    train_group_config = {}
    train_group_config['train'] = {}
    for i in arr:
        # group name
        if i_group % group_size == 0:
            group_id = 1 + int(i_group / group_size)
            group_name = 'BJCH_' + str(group_id)
            train_group_config['train'][group_name] = {}
        index = int(i_group % group_size + 1)
        sample_lesion_name = train_config['train']['lesion_' + str(i)]['sample_lesion_name']
        my_sample_lesion_name = str(index) + '_lesion_' + sample_lesion_name
        train_group_config['train'][group_name][my_sample_lesion_name] = {}
        train_group_config['train'][group_name][my_sample_lesion_name]['images_data_niix'] = \
            train_config['train']['lesion_' + str(i)][sample_lesion_name]['images_data_niix']
        train_group_config['train'][group_name][my_sample_lesion_name]['class_description'] = \
            train_config['train']['lesion_' + str(i)][sample_lesion_name]['class_description']
        i_group += 1

    target_root = grouped_subfolder
    with open(os.path.join(target_root, 'model_c_train_config.json'), 'w', encoding='utf-8') as f_json:
            json.dump(train_group_config, f_json, indent=4)

    # -----------------------------------------------------------------------------------
    # generate test config file
    predict_config = {}
    predict_config['predict'] = {}
    lesion_size=0
    i_group = 0
    group_size = 1000
    for test_group in test_groups:
        dcm_root = os.path.join(src_subfolder, test_group, 't2_post')
        for patient_root, subdirs, files in os.walk(dcm_root):
            if not 'labels.json' in files: continue
            patient_root_vec = patient_root.split(os.path.sep)
            aug_num = int(patient_root_vec[-1])
            if aug_num == 0:
                with open(os.path.join(patient_root, 'labels.json'), 'r', encoding='utf-8') as f_json:
                    label_config = json.load(f_json)
                    TRG_Rate = label_config['Label']['TRG']
                    TRG_Rate = label_config['Label']['downstage']
                    if TRG_Rate == 'NULL': continue
                    if int(TRG_Rate) > 0.5:
                        TRG_Rate = 'ab_yes_downstage'
                    elif int(TRG_Rate) < 0.5:
                        TRG_Rate = 'no_downstage'
                    else:continue
                lesion_size += 1
                if i_group % group_size == 0:
                    group_id = 1 + int(i_group / group_size)
                    group_name = 'BJCH_' + str(group_id)
                    predict_config['predict'][group_name] = {}
                sample_name = test_group+ '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                              + '_' +patient_root_vec[-2]+ '_Aug_' + patient_root_vec[-1]
                predict_config['predict'][group_name][sample_name] ={}

                phase_files = []
                image_file = os.path.join(patient_root, 'lesion_patch.nii.gz')
                for phase in phases:
                    phase_name = image_file.replace('t2_post', phase)
                    if 'Dapp' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Dapp_lesion_patch_1')
                    elif 'Kapp' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Kapp_lesion_patch_1')
                    elif 'Blog1000' in phase:
                        phase_name = phase_name.replace('lesion_patch_1', 'Blog1000_lesion_patch_1')
                    phase_files.append(phase_name)
                predict_config['predict'][group_name][sample_name]['images_data_niix'] = phase_files
                predict_config['predict'][group_name][sample_name]['reference_class_description'] = TRG_Rate
                predict_config['predict'][group_name][sample_name]['class_description'] = ''
                i_group += 1

    target_root = grouped_subfolder
    with open(os.path.join(target_root, 'model_c_predict_config.json'), 'w', encoding='utf-8') as f_json:
            json.dump(predict_config, f_json, indent=4)

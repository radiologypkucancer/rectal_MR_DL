# ----------------------------------------------------------------------------------
# imports
from data_preparation.convert_niix_to_npz import DataType
from data_preparation.convert_niix_to_npz import add_np_data

import json, os

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # data config files root
    src_subfolder = 'D:\\data\\sunyingshi\\cross_grouped_data_for_3kinds_label_Kapp_Dapp_full_lesion_0410'
    project_data_config_files = os.path.join(src_subfolder, 'model_configs_test', \
                                     'model_C_Tdegrade_abyes_no_with_blog1000_data')
    # model root
    model_root = 'D:\\data\\ailab\\Rectal_MR\\models\\BJCH_rectal_MR_model_c_test'

    # model settings
    niix_size = (128, 128, 16)
    classx_dict = {
        "0": "ab_yes_downstage",
        "1": "no_downstage"
    }

    # convert key from str into int
    new_classx_dict = {}
    for key in classx_dict.keys():
        new_classx_dict[int(key)] = classx_dict.get(key)

    # convert training data
    lesion_classification_train_config_file = os.path.join(project_data_config_files, 'model_c_train_config.json')
    with open(lesion_classification_train_config_file, 'r', encoding='utf-8') as f_json:
        train_config = json.load(f_json)
    if train_config is not None and train_config.__class__ is dict:
        train_groups = train_config.get('train')
        if train_groups.__class__ is not dict: logger.info('invalid train_config.')
        # adding new training data if it is not existed
        add_np_data(grouped_data_sets=train_groups, classx_dict=new_classx_dict, niix_size=niix_size, model_root=model_root,  \
                    data_type=DataType.TRAINING)

    # convert validation data
    lesion_classification_validation_config_file = os.path.join(project_data_config_files, 'model_c_val_config.json')
    with open(lesion_classification_validation_config_file, 'r', encoding='utf-8') as f_json:
        validation_config = json.load(f_json)
    if validation_config is not None and validation_config.__class__ is dict:
        validation_groups = validation_config.get('validation')
        # adding new validation data if it is not existed
        add_np_data(grouped_data_sets=validation_groups, classx_dict=new_classx_dict, niix_size= niix_size, model_root=model_root, \
                    data_type=DataType.VALIDATION)


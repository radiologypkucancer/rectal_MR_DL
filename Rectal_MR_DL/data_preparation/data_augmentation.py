# ----------------------------------------------------------------------------------
# imports
import SimpleITK as sitk
import json, os
from data_preparation.common_img_processing.augmentation import sitk_augment_by_rotate_roi_3d

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()
#-----------------------------------------------------------------------------------
#
def augment_3d_data(src_root, label_root, target_root):
    """
    :param src_root:
    :param target_root:
    :return:
    """
    if not os.path.exists(src_root): return False
    if not os.path.exists(target_root): os.mkdir(target_root)

    # recording of failed patient
    failed_config = {}
    failed_config['failed_aug_patient'] = {}
    failed_num = 0
    # walking though the patients
    for patient_root, dirs, files in os.walk(src_root):
        if not 'labels.json' in files: continue
        patient_root_vec = patient_root.split(os.path.sep)
        patient_num = patient_root_vec[-2]
        lesion_patch_num = patient_root_vec[-1]
        im_type = patient_root_vec[-3]
        logger.info('working on %s', patient_num)

        label_config = {}
        label_config['Label'] = {}

        nb_rotations = 29

        labels_root = os.path.join(label_root, 'adc_post',patient_num, 'lesion_patch_1')

        with open(os.path.join(labels_root, 'labels.json'), 'r', encoding='utf-8') as f_json:
            label_config0 = json.load(f_json)
            TRG_RATE = label_config0['Label']['TRG']
            if float(TRG_RATE) < 0.1:  nb_rotations = 119
            label_config['Label']  = label_config0['Label']

        temp_target_root = os.path.join(target_root, patient_num)
        if not os.path.exists(temp_target_root):
            os.makedirs(temp_target_root)
        temp_target_root = os.path.join(temp_target_root, lesion_patch_num)
        if not os.path.exists(temp_target_root):
            os.makedirs(temp_target_root)

        for file in files:
            if file.endswith('nii.gz') and 'main_bg_removed' in file: # main_bg_removed
                image_file = os.path.join(patient_root, file)
                break

        for file in files:
            if file.endswith('nii.gz') and 'lesion_mask' in file:
                label_file = os.path.join(patient_root, file)
                try:
                    sitk_image = [sitk.ReadImage(os.path.join(image_file)) ]
                    label = sitk.ReadImage(os.path.join(label_file))
                    vols, labels = sitk_augment_by_rotate_roi_3d(sitk_image,
                                                                 label,
                                                                 nb_rotations=nb_rotations)
                    for i, rot in enumerate(vols):
                        root = os.path.join(temp_target_root, str(i))
                        if not os.path.exists(root):
                            os.makedirs(root)
                        else:
                            continue
                        with open(os.path.join(root,'labels.json'), 'w', encoding='utf-8') as f_json:
                            json.dump(label_config, f_json, indent=4)
                        for j, p in enumerate(rot):
                            sitk.WriteImage(p, os.path.join(root, 'lesion_patch' + '.nii.gz'))
                except:
                    failed_num += 1
                    failed_config['failed_aug_patient'][str(failed_num)] = patient_num + '_' + lesion_patch_num
                    logger.warning(patient_num, file)
                break

    with open(os.path.join(target_root,'augment_sliced_data_failed.json'), 'w', encoding='utf-8') as f_json:
        json.dump(failed_config, f_json, indent=4)

    return True

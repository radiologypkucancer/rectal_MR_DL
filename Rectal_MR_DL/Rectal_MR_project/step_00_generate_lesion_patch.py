# ----------------------------------------------------------------------------------
# imports
import os, json
import SimpleITK as sitk
from data_preparation.common_img_processing.labels import get_foreground_labels_list_for_bjzl
from data_preparation.common_img_processing.cropping import crop_mask_region_and_generate_bbox

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

#-----------------------------------------------------------------------------------
#
def remove_backgroud(sitkImage, sitkLabel):
    img_array = sitk.GetArrayFromImage(sitkImage)
    img_label = sitk.GetArrayFromImage(sitkLabel)
    img_array_bg_removed = img_array * img_label
    bg_removed_sitkImage = sitk.GetImageFromArray(img_array_bg_removed)
    bg_removed_sitkImage.SetOrigin(sitkImage.GetOrigin())
    bg_removed_sitkImage.SetSpacing(sitkImage.GetSpacing())
    bg_removed_sitkImage.SetDirection(sitkImage.GetDirection())
    return bg_removed_sitkImage

#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # data root
    src_root = 'D:\\data\\sunyingshi\\T2train_nii\\test'

    # recording of failed patient
    failed_config = {}
    failed_config['failed_aug_patient'] = {}
    failed_num = 0

    for subroot, dirs, files in os.walk(src_root):
        patient_no = str(os.path.basename(subroot))
        if len(files) <= 2: continue
        if 'labels.json' in files: continue
        logger.info('working on %s', patient_no)

        for file in files:
            if file.endswith('nii.gz') and 'main' in file :
                image_name = file
                image_file = os.path.join(subroot,file)
            elif file.endswith('nii.gz')and 'mask' in file :
                label_name = file
                label_file = os.path.join(subroot,file)

        if label_file is None: continue
        label_config = {}
        label_config['Label'] = {}
        label_config['Label']['Imaging Modality'] = 'Rectal_MR'
        label_config['Label']['Focal Lesion Type'] = 'rectal tumor'
        label_config['Label']['YPCR'] = ''
        label_config['Label']['downstage'] = ''
        label_config['Label']['TRG'] = ''

        try:
            image = sitk.ReadImage(image_file)
            label = sitk.ReadImage(label_file)
            sitk_labels_list = get_foreground_labels_list_for_bjzl(label, reset_label_id=True)

            label_num = 0
            for sitk_label in sitk_labels_list:
                label_num += 1
                lesion_patch, bounding_box = crop_mask_region_and_generate_bbox(image, sitk_label,spare_boundary_mm=(6,6,6))
                bg_removed_sitkImage = remove_backgroud(sitkImage=image, sitkLabel=sitk_label)

                root = os.path.join(os.path.join(subroot, 'lesion_patch_' + str(label_num)))
                if not os.path.exists(root):
                    os.makedirs(root)
                else:
                    continue
                label_file_name = 'lesion_mask_' + str(label_num) + '.nii.gz'
                sitk.WriteImage(sitk_label, os.path.join(root, label_file_name))
                with open(os.path.join(root, 'labels.json'), 'w', encoding='utf-8') as f_json:
                    json.dump(label_config, f_json, indent=4)
                lesion_patch_file_name = 'lesion_patch_' + str(label_num) + '.nii.gz'
                bounding_box_file_name = 'bounding_box_' + str(label_num) + '.nii.gz'
                sitk.WriteImage(lesion_patch, os.path.join(root, lesion_patch_file_name))
                sitk.WriteImage(bounding_box, os.path.join(root, bounding_box_file_name))
                sitk.WriteImage(bg_removed_sitkImage, os.path.join(root, 'main_bg_removed' + '.nii.gz'))
                sitk.WriteImage(image, os.path.join(root, image_name))
                sitk.WriteImage(label, os.path.join(root, label_name))
        except:
            failed_num += 1
            failed_config['failed_aug_patient'][str(failed_num)] = patient_no

            logger.warning(patient_no)

    with open(os.path.join(src_root,'generate_lesion_patch_failed_1.json'), 'w', encoding='utf-8') as f_json:
            json.dump(failed_config, f_json, indent=4)
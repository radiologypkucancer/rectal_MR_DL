# ----------------------------------------------------------------------------------
# imports
import os
from data_preparation.data_augmentation import augment_3d_data

# ----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # data groups
    groups = {
        't2_post': 't2_post',
        't2_pre': 't2_pre',
        'Dapp_post': 'Dapp_post',
        'Dapp_pre': 'Dapp_pre',
        'Kapp_post': 'Kapp_post',
        'Kapp_pre': 'Kapp_pre',
        'Blog1000_post': 'Blog1000_post',
        'Blog1000_pre': 'Blog1000_pre'
    }
    # data and label root
    rawdata_root0 = "D:\\data\\sunyingshi\\selected_4kinds_data"
    label_root = "D:\\data\\sunyingshi\\selected_4kinds_data"
    target_root0 = "D:\\data\\sunyingshi\\auged_8_kinds_data_3d_volume_TRG"

    if not os.path.exists(target_root0): os.mkdir(target_root0)

    # prepare data
    for group in groups.keys():
        rawdata_root = os.path.join(rawdata_root0, groups.get(group))
        if not os.path.exists(rawdata_root):
            rawdata_root = rawdata_root.replace('selected_4kinds_data', 'selected_4kinds_data_kapp_dapp')
            if not os.path.exists(rawdata_root):
                rawdata_root = rawdata_root.replace('selected_4kinds_data_kapp_dapp', 'selected_Blog1000_data')
        target_root = os.path.join(target_root0, groups.get(group))
        if not os.path.exists(target_root): os.mkdir(target_root)
        # data augmentation
        augment_3d_data(rawdata_root, label_root, target_root)
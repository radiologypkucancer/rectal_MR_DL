# ------------------------------------------------------------------------------
# imports
import os
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':
    root = "D:\\data\\ailab\\Rectal_MR\\models\\BJCH_rectal_MR_model_c_test\\training_images_00001"
    im_file = "arr_0.npy"
    im = np.load(os.path.join(root,im_file))

    num = 8
    height = 128
    slice = np.zeros([16, height, height*num])
    #
    s =128
    #
    for k in range(0, num):
        lo = k * height
        hi = (k +1) * height
        slice[:, :, lo:hi] = im[s, k, ::]

    # plot 8th slice images
    slice16 = slice[7, :,:]
    plt.figure("slice")
    plt.imshow(slice16)
    plt.show()
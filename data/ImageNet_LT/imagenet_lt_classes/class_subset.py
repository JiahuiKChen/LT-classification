import numpy as np
from os import listdir
import shutil

# class labels for the classes wanted 
classes = [
    # 30 few classes: 
    # first 10 are items or structures/scenes (includes hard cases like "iron, 606")
    579, 606, 632, 702, 759, 718, 873, 866, 498, 776,
    # next 10 are animals
    397, 38, 15, 73, 102, 140, 142, 170, 260, 376,
    # last 10 are food or clothes or abstract
    568, 614, 916, 935, 959, 962, 963, 992, 738, 941,

    # 30 median classes
    # first 10 are animals
    359, 344, 377, 313, 273, 222, 63, 25, 6, 395,
    # next 10 are items or structures/scenes 
    750, 706, 703, 698, 690, 684, 838, 850, 919, 801,
    # last 10 are food or clothes or people 
    795, 983, 981, 982, 960, 697, 931, 928, 956, 967,

    # 30 many classes
    # first 10 are animals
    22, 29, 48, 108, 372, 388, 357, 394, 338, 341,
    # next 10 are items or structures/scenes
    404, 419, 460, 486, 540, 548, 567, 662, 712, 760, 
    # last 10 are food or clothes 
    443, 515, 578, 770, 961, 965, 655, 711, 903, 933
]
class_set = set(classes)

# generate train, val, and test txt files for given classes
# data_files = [
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_train.txt",
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_val.txt",
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_test.txt"
#     ]
# out_files = [
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_train_90.txt",
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_val_90.txt",
#     "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_test_90.txt" 
#     ]
# for i in range(len(data_files)):
#     output_str = ""
#     file = data_files[i]
#     out_file = out_files[i]
#     with open(file) as data:
#         # if the line contains an image of a class we want, write that line to our new file 
#         for line in data:
#             class_label = int(line.split()[1])
#             if class_label in class_set:
#                 output_str += line

#     with open(out_file, "w") as outuput_file:
#         outuput_file.write(output_str)

# move generated images of these classes 
source_dirs = [
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/rand_img_cond",
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/cutmix", 
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/mixup",
] # TODO: dropout once it's done 
destination_dirs = [
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/rand_img_cond_90",
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/cutmix_90", 
    "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/mixup_90",
]
for i in range(len(source_dirs)):
    source_dir = source_dirs[i]
    dest_dir = destination_dirs[i]
    all_imgs = listdir(source_dir)
    for img in all_imgs:
        label = int(img.split("_")[0])
        if label in class_set:
            shutil.copy(f"{source_dir}/{img}", f"{dest_dir}/{img}")

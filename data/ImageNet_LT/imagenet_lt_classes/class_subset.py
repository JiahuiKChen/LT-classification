import numpy as np
from os import listdir
import shutil

# class labels for the classes wanted 
# 30 many-classes
classes = [
    # first 10 are animals
    50, 107, 67, 148, 207, 244, 367, 345, 302, 333, 

    # next 10 are objects, buildings/places, or vehicles
    414, 417, 441, 519, 466, 509, 555, 449, 548, 532,

    # last 10 clothes or food
    578, 608, 652, 679, 934, 987, 898, 924, 929, 658
]


# 30 class subset: 10-12 hour resnext18 train time 
# classes = [
#     # 10 few classes
#     # first 4 are items or structures/scenes (includes hard cases like "iron, 606")
#     # grand piano, iron, pier, "cinema, movie theater, movie theatre, movie house, picture palace"
#     579, 606, 718, 498, 
#     # next 3 are animals: pufferfish, "banded gecko", robin, 
#     397, 38, 15,
#     # last 3 are: fur coat, website, carbonara
#     568, 916, 959,

#     # 10 median classes
#     # first 4 are animals: hippo, marmoset, walking stick, dingo
#     344, 377, 313, 273, 
#     # next 3: quilt, park bench, "sunscreen, sunblock, sun blocker"
#     750, 703, 838,
#     # last 3: ski, scuba diver, "ballplayer, baseball player"
#     795, 983, 981,

#     # 10 many classes
#     # first 3 are animals: "green snake, grass snake", sea anemone, llama
#     55, 108, 355,
#     # next 3 (items or structures/scenes): "breakwater, groin, groyne, mole, bulwark, seawall, jetty", cello, "drilling platform, offshore rig"  
#     460, 486, 540,
#     # last 4 are food or clothes: cowboy hat, gown, running shoe, burrito
#     515, 578, 770, 965
# ]

# 90 class subset: ~24 hour resnext18 train time
# classes = [
#     # 30 few classes: 
#     # first 10 are items or structures/scenes (includes hard cases like "iron, 606")
#     579, 606, 632, 702, 759, 718, 873, 866, 498, 776,
#     # next 10 are animals
#     397, 38, 15, 73, 102, 140, 142, 170, 260, 376,
#     # last 10 are food or clothes or abstract
#     568, 614, 916, 935, 959, 962, 963, 992, 738, 941,

#     # 30 median classes
#     # first 10 are animals
#     359, 344, 377, 313, 273, 222, 63, 25, 6, 395,
#     # next 10 are items or structures/scenes 
#     750, 706, 703, 698, 690, 684, 838, 850, 919, 801,
#     # last 10 are food or clothes or people 
#     795, 983, 981, 982, 960, 697, 931, 928, 956, 967,

#     # 30 many classes
#     # first 10 are animals
#     22, 29, 48, 108, 372, 388, 357, 394, 338, 341,
#     # next 10 are items or structures/scenes
#     404, 419, 460, 486, 540, 548, 567, 662, 712, 760, 
#     # last 10 are food or clothes 
#     443, 515, 578, 770, 961, 965, 655, 711, 903, 933
# ]
class_set = set(classes)

# generate train, val, and test txt files for given classes
# A40: //datastor1/jiahuikchen/
# MIDI: /datastor1/jiahuikchen/tmix 
data_files = [
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train.txt",
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_val.txt",
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_test.txt"
    ]
out_files = [
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train_30_many.txt",
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_val_30_many.txt",
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_test_30_many.txt" 
    ]
for i in range(len(data_files)):
    output_str = ""
    file = data_files[i]
    out_file = out_files[i]
    with open(file) as data:
        # if the line contains an image of a class we want, write that line to our new file 
        for line in data:
            class_label = int(line.split()[1])
            if class_label in class_set:
                output_str += line

    with open(out_file, "w") as outuput_file:
        outuput_file.write(output_str)

# move generated images of these classes 
source_dirs = [
    "/datastor1/jiahuikchen/synth_ImageNet/rand_img_cond",
    "/datastor1/jiahuikchen/synth_ImageNet/cutmix", 
    "/datastor1/jiahuikchen/synth_ImageNet/mixup",
    "/datastor1/jiahuikchen/synth_ImageNet/dropout",
]  
destination_dirs = [
    "/datastor1/jiahuikchen/synth_ImageNet/rand_img_cond_30_many",
    "/datastor1/jiahuikchen/synth_ImageNet/cutmix_30_many", 
    "/datastor1/jiahuikchen/synth_ImageNet/mixup_30_many",
    "/datastor1/jiahuikchen/synth_ImageNet/dropout_30_many",
]
for i in range(len(source_dirs)):
    source_dir = source_dirs[i]
    dest_dir = destination_dirs[i]
    all_imgs = listdir(source_dir)
    for img in all_imgs:
        label = int(img.split("_")[0])
        if label in class_set:
            shutil.copy(f"{source_dir}/{img}", f"{dest_dir}/{img}")

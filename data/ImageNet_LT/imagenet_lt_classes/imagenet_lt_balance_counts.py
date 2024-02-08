import ast
from os import listdir, remove, path
import shutil

# dictionary with label (int) : class text (string)
with open("imagenet1000_class_labels.txt") as data:
    class_labels = ast.literal_eval(data.read())
# print(class_labels)
    
# count how many of each class we have in ImageNetLT train set 
class_counts = {}
with open("/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_train_30.txt") as training_data:
    for line in training_data:
        class_label = int(line.split()[1])
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1

total_images_needed = 0
for c in class_counts:
    total_images_needed += 1280 - class_counts[c]

# save file where each line is:
#   <class label (int)> <text label> <count of images to generate to get to 1280>
output_str = ""
# first_273_count = 0
# after_273_count = 0
sanity = 0
for label in class_counts:
    int_label = int(label)
    txt_label = class_labels[label]
    needed_count = 1280 - class_counts[int_label]
    sanity += needed_count
    line = f"{int_label} \"{txt_label}\" {needed_count}\n"
    output_str += line
    # if label <= 273:
    #     first_273_count += needed_count
    # else:
    #     after_273_count += needed_count

# with open("imagenet_lt_balance_counts_30subset.txt", "w") as outuput_file:
#     outuput_file.write(output_str)

# 1,164,154 total images needed RIPPPP
print(total_images_needed)
print(sanity == total_images_needed)
# print(f"Classes 0-273 need: {first_273_count}")
# print(f"Classes 238-999 need: {after_273_count}")
    
# check dropout (it's gud) 
# dropout = listdir("/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/dropout/")
# dropout_counts = {}
# for img in dropout:
#     label = int(img.split("_")[0])
#     if label not in dropout_counts:
#         dropout_counts[label] = 1
#     else:
#         dropout_counts[label] += 1

# missing_cause273 = 0
# for label in class_counts:
#     if label in dropout_counts:
#         if (1280 - class_counts[label]) != dropout_counts[label]:
#             print(f"Class {label} has {dropout_counts[label]} images \t should be {1280 - class_counts[label]}")
#     else:
#         missing_cause273 += 1

# print(missing_cause273)

import ast
from os import listdir, remove, path

####### finds (and creates file) of:
# <class label (int)> <text label> <count of images to generate to get to 1280> 
# given a training file

# dictionary with label (int) : class text (string)
with open("/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/imagenet_lt_classes/imagenet1000_class_labels.txt") as data:
    class_labels = ast.literal_eval(data.read())
# print(class_labels)
    
# count how many of each class we have in ImageNetLT train set 
class_counts = {}
with open("/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train_30_many_to_median.txt") as training_data:
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
sanity = 0
for label in class_counts:
    int_label = int(label)
    txt_label = class_labels[label]
    needed_count = 1280 - class_counts[int_label]
    sanity += needed_count
    line = f"{int_label} \"{txt_label}\" {needed_count}\n"
    output_str += line

with open("imagenet_lt_balance_counts_30_many_to_median.txt", "w") as outuput_file:
    outuput_file.write(output_str)

# 1,164,154 total images needed RIPPPP (for all 1k classes)
print(total_images_needed)
print(sanity == total_images_needed)
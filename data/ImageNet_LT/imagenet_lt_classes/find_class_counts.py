import ast
import numpy as np

# greater than 100 images is MANY
many_count_thresh = 100
# less than 20 is FEW 
few_count_thresh = 20
# median is anything in between (inclusive)

# dictionary with label (int) : class text (string)
with open("imagenet1000_class_labels.txt") as data:
    class_labels = ast.literal_eval(data.read())

# count how many of each class we have in ImageNetLT train set 
class_counts = {}
# also track what "n...." label the integer label has 
class_folders = {}
with open("/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_train.txt") as training_data:
    for line in training_data:
        class_label = int(line.split()[1])
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1
            img_path = line.split()[0]
            class_folders[class_label] = img_path.split("/")[1] 

# go through all classes and categorize them according to counts
few = []
median = []
many =[]
for class_label in class_counts:
    if class_counts[class_label] < few_count_thresh:
        few.append(class_label)
    elif class_counts[class_label] > many_count_thresh:
        many.append(class_label)
    else:
        median.append(class_label)

print(f"{len(few)} classes with < 20 images [FEW]")
print(f"{len(median)} classes with between 20-100 images [MEDIAN]") 
print(f"{len(many)} classes with > 100 images [MANY]")

# for each class count category, save file where each line is:
#   <class label (int)> <text label> <n... folder> <count of images in train>
names = ["few", "median", "many"]
categories = [few, median, many] 
for i in range(len(categories)):
    category = categories[i]
    name = names[i]
    output_str = ""
    for label in category:
        line = f"{label} \"{class_labels[label]}\" {class_folders[label]} {class_counts[label]}\n"
        output_str += line
    with open(f"imagenet_lt_{name}_train.txt", "w") as outuput_file:
        outuput_file.write(output_str)
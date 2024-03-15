import json

# given a training data txt file, write a JSON dict of:
# class label: count of images
# where the class label is [0, num_classes] (for subsets this is what the labels are)
expected_class_num = 10
train_file = "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train_30_many_to_median.txt"
output_dir = "/datastor1/jiahuikchen/LT-classification/cls_freq"
json_name = "ImageNet_LT_train_30_many_to_median_only.json"

class_counts = [0] * expected_class_num
# used to map classes to [0, num_classes]
class_map = {}
class_ind = 0
with open(train_file) as training_data:
    for line in training_data:
        class_label = int(line.split()[1])
        if class_label not in class_map:
            class_map[class_label] = class_ind
            class_ind += 1
            class_counts[class_map[class_label]] = 1
        else:
            class_counts[class_map[class_label]] += 1

with open(f"{output_dir}/{json_name}", "w") as out_file:
    json.dump(class_counts, out_file)
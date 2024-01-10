import ast

# dictionary with label (int) : class text (string)
with open("imagenet1000_class_labels.txt") as data:
    class_labels = ast.literal_eval(data.read())
# print(class_labels)
    
# count how many of each class we have in ImageNetLT train set 
class_counts = {}
with open("ImageNet_LT/ImageNet_LT_train.txt") as training_data:
    for line in training_data:
        class_label = int(line.split()[1])
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1

# save file where each line is:
#   <class label (int)> <text label> <count of images to generate to get to 1280>
output_str = ""
for label in class_labels:
    int_label = int(label)
    txt_label = class_labels[label]
    needed_count = 1280 - class_counts[int_label]
    line = f"{int_label} \"{txt_label}\" {needed_count}\n"
    output_str += line

with open("imagenet_lt_balance_counts.txt", "w") as outuput_file:
    outuput_file.write(output_str)

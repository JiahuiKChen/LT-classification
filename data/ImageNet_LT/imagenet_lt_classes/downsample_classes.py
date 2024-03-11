import random 

# class labels for the classes to downsample 
# 30 many-classes to downsample the number of REAL training images
classes = [
    # 10 many classes
    # first 3 are animals: "green snake, grass snake", sea anemone, llama
    55, 108, 355,
    # next 3 (items or structures/scenes): "breakwater, groin, groyne, mole, bulwark, seawall, jetty", cello, "drilling platform, offshore rig"  
    460, 486, 540,
    # last 4 are food or clothes: cowboy hat, gown, running shoe, burrito
    515, 578, 770, 965
]
class_set = set(classes)

# randomly sample to 10 (few) or 40 (median) real training images for these classes
TRAIN_IMG_COUNT = 40 
TRAIN_CLASS = "median"
TRAIN_DATA_TXT = "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train.txt" 
TRAIN_OUT_FILE = f"/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_train_30_many_to_{TRAIN_CLASS}.txt"

# read in dict of all images mapped to class
TRAIN_DICT = {}
with open(TRAIN_DATA_TXT) as train_file:
    for line in train_file:
        info = line.split() 
        class_label = int(info[1])
        img_path = info[0]

        if class_label in TRAIN_DICT:
            TRAIN_DICT[class_label].append(img_path)
        else:
            TRAIN_DICT[class_label] = [img_path]
# downsample classes in class_set to TRAIN_IMG_COUNT number of images 
downsampled_train = {}
for c in class_set:
    images = TRAIN_DICT[c]
    downsampled_imgs = random.sample(images, TRAIN_IMG_COUNT)
    downsampled_train[str(c)] = downsampled_imgs

# generate train txt file from downsampled train dict
train_str = ""
for c in downsampled_train.keys():
    img_paths = downsampled_train[c]
    for img_path in img_paths:
        img_line = f"{img_path} {c}\n"
        train_str += img_line
with open(TRAIN_OUT_FILE, "w") as outuput_file:
    outuput_file.write(train_str)

# generate val, and test txt files for given classes without downsampling
data_files = [
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_val.txt",
    "/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_test.txt"
    ]
out_files = [
    f"/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_val_30_many_to_{TRAIN_CLASS}.txt",
    f"/datastor1/jiahuikchen/LT-classification/data/ImageNet_LT/ImageNet_LT_test_30_many_to_{TRAIN_CLASS}.txt" 
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
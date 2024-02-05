import shutil
import os

# for each of the subsets' val and test txt files,
# move all these images into their own directories 
txt_files = [
    "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_test_30.txt", 
    "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_val_30.txt",
    "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_test_90.txt",
    "/mnt/zhang-nas/jiahuic/LT-classification/data/ImageNet_LT/ImageNet_LT_val_90.txt"
            ]
data_root = '/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012'
new_dirs_root = "/mnt/zhang-nas/jiahuic/ImageNetLT_subset_val_test"

for txt_file in txt_files:
    name = txt_file.split("/")[-1].strip(".txt")
    with open(txt_file) as img_file:
        for line in img_file:
            if "test" in name:
                # test data in imagenet data dirs is just val/*.JPEG, txt files have another dir
                txt_path = line.split()[0].split("/")
                local_path = os.path.join(txt_path[0], txt_path[-1])
            else:
                local_path = line.split()[0]
            img_path = os.path.join(data_root, local_path)

            # mkdir if it doesn't exist
            dest_dir = os.path.join(new_dirs_root, name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            # copy image to corresponding subdir 
            shutil.copy(img_path, f"{dest_dir}/{img_path.split('/')[-1]}")
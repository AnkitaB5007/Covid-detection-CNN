import os
import datetime
import numpy as np
import cv2
import shutil


def transform_image(img):
    min_size = min(img.shape[0],img.shape[1])
    max_crop = min_size - 224       # 224 for ResNet50

    pil_transform = transforms.ToPILImage()
    resize_transform = transforms.Resize(224)

    total_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2),
            transforms.Pad((10,10))
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomRotation(30),
        transforms.RandomCrop(min_size - round(max_crop/10))
    ])

    image = pil_transform(img)

    if min_size < 224:
        image = resize_transform(image)

    return total_transform(image)

def data_augmentation(workspace, data_dir, source_dirs):
    augset_dir = os.path.join(workspace, 'Augmented_TrainSet')
    if os.path.isdir(augset_dir) != True:
        os.mkdir(augset_dir)
    for c in source_dirs:
        if (os.path.isdir(os.path.join(augset_dir, c)) != True):
            os.mkdir(os.path.join(augset_dir, c))
        imgs = [x for x in os.listdir(os.path.join(data_dir, c))]
        for i, img in enumerate(imgs):
            original_img = img
            source_path = os.path.join(data_dir, c, original_img)
            target_path = os.path.join(augset_dir, c)
            shutil.copy(source_path, target_path)
            img = cv2.imread(source_path)
            for j in range(12):
                new_img = np.array(transform_image(img))
                new_img_name = "{}_copy{}.{}".format("".join(original_img.split(".")[:-1]),j,original_img.split(".").pop(-1))
                cv2.imwrite(os.path.join(target_path, new_img_name), new_img)


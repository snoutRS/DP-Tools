import shutil
import sys
import cv2
import os
import random
import numpy as np
import glob

# TODO 文件地址生成器(数据集位置（地址字符串）)
def count_files(inPath):
    files = []
    labels = []
    subdirs = os.listdir(inPath)
    subdirs.sort()
    for index in range(len(subdirs)):
        subdir = os.path.join(inPath, subdirs[index])
        sys.stdout.flush()
        for image_path in glob.glob("{}/*.jpg".format(subdir)):
            files.append(image_path)
            labels.append(index)
    return files, labels, len(subdirs)


# TODO 读取图片cv2.imdecode防止中文名称导致图片报错（地址字符串）
def img_read(file_path):
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return image


# TODO 数据清洗(需要清洗的数据集目录（地址字符串），不合格图片目录（地址字符串）)
def clean(inPath,dryPath):
    dataset = inPath  # 数据集路径
    labels = os.listdir(dataset)  # 图片类型
    for label in labels:
        # 清洗失败和不符合要求的图片
        label_path = os.path.join(dataset, label)
        images = os.listdir(label_path)
        for order, img in enumerate(images):  # 获得文件名和序号

            # 获取图像文件的全路径
            img_path = os.path.join(label_path, img)

            # 读取图像文件,如果失败,则进入下一个循环
            img = img_read(img_path)
            if img is None:
                print('This picture cannot be opened: ', img_path)
                shutil.move(img_path, dryPath)
                continue
            if img.ndim < 3:
                print('This picture has an incorrect number of channels: ', img_path)
                shutil.move(img_path, dryPath)


# TODO 图片加载(地址字符串,float,float)
def load_image(image_path,width=32,hight=32,Dtype="float32"):
    size = (width, hight)
    # 根据路径加载图片
    img = img_read(image_path)
    #  尺寸调整，调整后为读入尺寸
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
    # 图片数据像素点值转化为浮点型
    img = img.astype(Dtype)
    # 归一化
    img /= 255.
    return np.array(img)

# TODO NPY封装(训练比率（0-1）,输出文件地址（图片地址字符串）)
def encapsulationDATA(inPath,OUTPUT_DATA,train_ratio=0.8):
    files, labels, len_tu = count_files(inPath)

    c = list(zip(files, labels))
    random.shuffle(c)
    files, labels = zip(*c)
    files = list(files)
    labels = np.array(labels)

    file_set = []
    for i in range(len(files)):
        ff = load_image(files[i])
        file_set.append(ff)
    file_set = np.array(file_set)

    train_size = int(len(files) * train_ratio)
    print("train_size: ", train_size)
    train_X = file_set[:train_size]
    train_Y = labels[:train_size]
    test_X = file_set[train_size:]
    test_Y = labels[train_size:]

    print(len(train_X), len(train_Y), len(test_X), len(test_Y))
    processed_data = np.asarray([train_X, train_Y, test_X, test_Y])
    np.save(OUTPUT_DATA, processed_data)

# TODO Fit_gen生成器（（地址列表），标签（列表），batch（批次数字））
def fit_gen(files_r, labels_r, batch=32, label="label"):
    start = 0

    while start < len(files_r):
        stop = start + batch
        if stop > len(files_r):
            stop = len(files_r)
        imgs = []
        lbs = []
        for i in range(start, stop):
            imgs.append(load_image(files_r[i]))
            lbs.append(labels_r[i])

        yield (np.array(imgs), np.array(lbs))

        if start + batch < len(files_r):
            start += batch
        else:
            c = list(zip(files_r, labels_r))
            random.shuffle(c)
            files_r, labels_r = zip(*c)
            start = 0


# TODO 调整尺寸（图片地址（地址字符串）,宽（float/int）,高（float/int））
def rdnsize(img, height, width):
    h, w, d = img.shape
    if h < height or w < width:
        result = cv2.resize(img, (width, height))
    else:
        y = random.randint(0, h-height)
        x = random.randint(0, w-width)
        result = img[y:y+height, x:x+width, :]
    return result


# TODO 随机翻转（图片地址（地址字符串）
def rdnflip(img):
    list1 = [0, 1, -1]
    sample = random.sample(list1, 1)
    result = cv2.flip(img, sample[0])  # 1水平翻转, 0垂直翻转，-1水平垂直翻转
    return result

# TODO 给值翻转（图片地址（地址字符串）,旋转方式（1水平翻转, 0垂直翻转, -1水平垂直翻转））
def flips(img,i):
    result = cv2.flip(img, i)
    return result

# TODO 随机旋转（图片地址（地址字符串）
def rotation(img):
    rows, cols, channel = img.shape
    r = random.randint(0, 10)
    changedaxiao = random.uniform(0.5, 1.5)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), r, changedaxiao)  # 旋转随机角度r，保持图片比例不变
    result = cv2.warpAffine(img, M, (cols, rows))
    return result

# TODO 给值旋转（图片地址（地址字符串）,旋转值（float/int））
def rotate(img,r):
    rows, cols, channel = img.shape
    changedaxiao = random.uniform(0.5, 1.5)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), r, changedaxiao)  # 旋转随机角度r，保持图片比例不变
    result = cv2.warpAffine(img, M, (cols, rows))
    return result

# TODO 均衡化（图片地址（地址字符串））
def equalize(img):
    (b, g, r) = cv2.split(img)
    b_equalize = cv2.equalizeHist(b)
    g_equalize = cv2.equalizeHist(g)
    r_equalize = cv2.equalizeHist(r)
    result = cv2.merge((b_equalize, g_equalize, r_equalize))
    return result

# TODO 随机添加亮度（图片地址（地址字符串）,最大亮度（float/int））
def rdnbright(img, max):
    rdn = random.randint(0, max)
    (h, w) = img.shape[:2]
    bright = np.ones((h, w), dtype=np.uint8)*rdn
    bright_bgr = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
    result = cv2.add(img, bright_bgr)
    return result

# TODO 随机添加噪声（图片地址（地址字符串）,均值(float/int),方差(float/int)）
def addnoise(img, mean, sign):
    (h, w) = img.shape[:2]
    noise = np.zeros((h, w), dtype=np.uint8)
    cv2.randn(noise, mean, sign)
    noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    result = cv2.add(img, noise_bgr)
    return result
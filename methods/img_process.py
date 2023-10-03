from glob import glob
from numpy import asarray
from tqdm import tqdm
import cv2


def img_process():
    img_path = "/Users/utilisateur/PycharmProjects/ImageTesto/img_data/**/"
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for img_type in ["*.png", "*.jpg"]:

        files = glob(pathname=f"{img_path}{img_type}", recursive=True)
        cmpt_test_l = 1
        cmpt_test_s = 1
        cmpt_train_l = 1
        cmpt_train_s = 1

        for file in tqdm(files):
            img = cv2.imread(filename=file, flags=0)
            img = cv2.resize(src=img, dsize=(250, 250), interpolation=cv2.INTER_AREA)

            if "test" in file:
                if "spoof" in file and cmpt_test_s <= 2500:
                    cmpt_test_s += 1
                    x_test.append(img)
                    y_test.append([1.])
                elif "live" in file and cmpt_test_l <= 2500:
                    cmpt_test_l += 1
                    x_test.append(img)
                    y_test.append([0.])
            elif "train" in file:
                if "spoof" in file and cmpt_train_s <= 7500:
                    cmpt_train_s += 1
                    x_train.append(img)
                    y_train.append([0.])
                elif "live" in file and cmpt_train_l <= 7500:
                    cmpt_train_l += 1
                    x_train.append(img)
                    y_train.append([1.])

            if cmpt_test_l == 2501 and cmpt_test_s == 2501:
                break
            elif cmpt_train_l == 7501 and cmpt_train_s == 7501:
                break

    x_train = asarray(x_train)
    y_train = asarray(y_train)
    x_test = asarray(x_test)
    y_test = asarray(y_test)

    return x_train, y_train, x_test, y_test

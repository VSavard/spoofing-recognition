from glob import glob
from PIL import Image
from tqdm import tqdm
from os import remove

img_path = "/Users/utilisateur/PycharmProjects/ImageTesto/img_data/**/"
files = glob(pathname=f"{img_path}*.png", recursive=True)

for file in tqdm(files):
    try:
        Image.open(fp=file, mode="r").convert("L")
    except:
        remove(file)


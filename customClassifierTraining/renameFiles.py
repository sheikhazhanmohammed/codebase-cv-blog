import os
import cv2
from tqdm import tqdm

i = 0
for fileName in tqdm(os.listdir("./data/dragonfly")):
    fileName = os.path.join("./data/dragonfly",fileName)
    try:
        image = cv2.imread(fileName)
        imageNewName = "image"+str(i)+".jpg"
        imageNewName = os.path.join("./data/dragonfly",imageNewName)
        cv2.imwrite(imageNewName, image)
        os.remove(fileName)
        i = i + 1
    except:
        os.remove(fileName)
import shutil
import random
import os

path = "./TrafficSignLight9-daytime/images/train/"
path_rain = "./TrafficSignLight9-daytime/images/trainrain/"
path_snow = "./TrafficSignLight9-daytime/images/trainsnow/"
path_fog = "./TrafficSignLight9-daytime/images/trainfog/"
path_night = "./TrafficSignLight9-daytime/images/trainnight/"

path_lb = "./TrafficSignLight9-daytime/labels/train/"
path_rain_lb = "./TrafficSignLight9-daytime/labels/trainrain/"
path_snow_lb = "./TrafficSignLight9-daytime/labels/trainsnow/"
path_fog_lb = "./TrafficSignLight9-daytime/labels/trainfog/"
path_night_lb = "./TrafficSignLight9-daytime/labels/trainnight/"

path_new = "./TrafficSignLight/images/train/"

path_new_lb = "./TrafficSignLight/labels/train/"

image_names = os.listdir(path)
length = len(image_names)

for idx, image_name in enumerate(image_names):
    rd = random.randint(1, 5)
    if rd == 1:
        shutil.copy(os.path.join(path, image_name), os.path.join(path_new, image_name))
        shutil.copy(os.path.join(path_lb, image_name.split(".")[0] + ".txt"), os.path.join(path_new_lb, image_name.split(".")[0] + ".txt"))
    elif rd == 2:
        shutil.copy(os.path.join(path_rain, image_name), os.path.join(path_new, image_name))
        shutil.copy(os.path.join(path_rain_lb, image_name.split(".")[0] + ".txt"), os.path.join(path_new_lb, image_name.split(".")[0] + ".txt"))
    elif rd == 3:
        shutil.copy(os.path.join(path_snow, image_name), os.path.join(path_new, image_name))
        shutil.copy(os.path.join(path_snow_lb, image_name.split(".")[0] + ".txt"), os.path.join(path_new_lb, image_name.split(".")[0] + ".txt"))
    elif rd == 4:
        new_image_name = image_name.split(".")[0] + "_fog." + image_name.split(".")[-1]
        shutil.copy(os.path.join(path_fog, new_image_name), os.path.join(path_new, new_image_name))
        shutil.copy(os.path.join(path_fog_lb, new_image_name.split(".")[0] + ".txt"), os.path.join(path_new_lb, new_image_name.split(".")[0] + ".txt"))
    elif rd == 5:
        new_image_name = image_name.split(".")[0] + "_night." + image_name.split(".")[-1]
        shutil.copy(os.path.join(path_night, new_image_name), os.path.join(path_new, new_image_name))
        shutil.copy(os.path.join(path_night_lb, new_image_name.split(".")[0] + ".txt"), os.path.join(path_new_lb, new_image_name.split(".")[0] + ".txt"))
    print(idx)
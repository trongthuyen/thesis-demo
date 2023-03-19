import cv2 as cv
from resnet_152 import resnet152_model
from constants import img_width, img_height, num_channels, num_classes
from lib_detection import load_model
from constants import char_list


def load_model_resnet152():
    model_weights_path = './models/model_resnet152/model.96-0.89.hdf5'
    model = resnet152_model(
        img_rows=img_height,
        img_cols=img_width,
        color_type=num_channels,
        num_classes=num_classes
    )
    model.load_weights(model_weights_path, by_name=True)
    return model

def load_model_wpod_net():
    wpod_net_path = "wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)
    return wpod_net

def draw_str(frame, position, text):
    x, y = position
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_PLAIN,
               1.5, (0, 0, 255), lineType=cv.LINE_AA, thickness=2)

def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString
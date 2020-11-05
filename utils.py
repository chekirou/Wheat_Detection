import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
def loade_file()
    file = pd.read_csv("/content/drive/My Drive/wheat-detection/global-wheat-detection/train.csv")
    bbox_items = file.bbox.str.split(',', expand=True)
    file['bbox_xmin'] = bbox_items[0].str.strip('[ ').astype(float)
    file['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)
    file['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)
    file['bbox_height'] = bbox_items[3].str.strip(' ]').astype(float)
    file.drop(columns=["bbox"], inplace = True)
    file["bbox_xmin"]= file["bbox_xmin"].astype("int64")
    file["bbox_ymin"]= file["bbox_ymin"].astype("int64")
    file["bbox_width"]= file["bbox_width"].astype("int64")
    file["bbox_height"]= file["bbox_height"].astype("int64")
    return file

def target(df, title):
    """ generate goal image from df of bounding boxes """
    blanc = np.zeros((1024, 1024))
    for _, row in df[df["image_id"] == title].iterrows():
        blanc[row.bbox_ymin   : row.bbox_ymin + row.bbox_height, row.bbox_xmin : row.bbox_xmin + row.bbox_width] = 1
    return blanc

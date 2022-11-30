#create and store dataset
import pandas as pd
import numpy as np
import cv2 as cv
import math
from spherical_geometry import polygon
from Utils import *
from nfov import *

HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
HEIGHT=1920
WIDTH=3840

_fov_points = dict()
_fov_polys = dict()


X1Y0Z0 = np.array([1, 0, 0])
_fov_x1y0z0_fov_points_euler = np.array([
    eulerian_in_range(-HOR_MARGIN, VER_MARGIN),
    eulerian_in_range(HOR_MARGIN, VER_MARGIN),
    eulerian_in_range(HOR_MARGIN, -VER_MARGIN),
    eulerian_in_range(-HOR_MARGIN, -VER_MARGIN)
])
_fov_x1y0z0_points = np.array([
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[0]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[1]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[2]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[3])
])



#david.create_and_store_sampled_dataset()
import Read_Dataset as david
#read from dataset
def load_data():
    dataset = david.load_sampled_dataset()
    # df with (dataset, user, video, times, traces)
                # times has only time-stamps
                # traces has only x, y, z (in 3d coordinates)
    data = [('david',
            'david' + '_' + user,
            'david' + '_' + video,
            dataset[user][video]
            ) for user in dataset.keys() for video in dataset[user].keys()]

    tmpdf = pd.DataFrame(data, columns=[
                    'ds', 'ds_user', 'ds_video',
                    'traces'])
    return tmpdf
#get xyz corner of the view port


def get_one_trace() -> np.array:
    dataset = load_data()
    return dataset.iloc[0]['traces'][0]

trace = get_one_trace()
print(trace)


def xyz2uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)

def fov_points(trace) -> np.ndarray:
    if (trace[1], trace[2], trace[3]) not in _fov_points:
        rotation = rotationBetweenVectors(X1Y0Z0, np.array([trace[1],trace[2],trace[3]]))
        #find 3d corners
        points = np.array([
            rotation.rotate(_fov_x1y0z0_points[0]),
            rotation.rotate(_fov_x1y0z0_points[1]),
            rotation.rotate(_fov_x1y0z0_points[2]),
            rotation.rotate(_fov_x1y0z0_points[3]),
        ])
        #convert into 2d corners
        points_converted = np.array([
                xyz2uv(points[0]),
                xyz2uv(points[1]),
                xyz2uv(points[2]),
                xyz2uv(points[3])
            ])
            
        _fov_points[(trace[1], trace[2], trace[3])] = points_converted
    return _fov_points[(trace[1], trace[2], trace[3])]

corners= fov_points(trace)
print(corners)

def uv2coor(uv):
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * WIDTH- 0.5
    coor_y = (-v / np.pi + 0.5) * HEIGHT - 0.5
    return np.concatenate([coor_x, coor_y], axis=-1)

def fov_coor(corners):
    fov_coor = np.array([
        uv2coor(corners[0]),
        uv2coor(corners[1]),
        uv2coor(corners[2]),
        uv2coor(corners[3])
    ])
    return fov_coor

coors = fov_coor(corners)
print(coors)

def crop_image(img,coors):
    x,y = np.split(coors[1],2,axis=-1)
    xx,yy = np.split(coors[3],2,axis=-1)

    crop_img = img[int(y):int(yy),int(x):int(xx)]
    cv.imshow('cropped',crop_img)
    return crop_img



img =cv.imread('./images/frame at 1 th second.jpg')
crop = crop_image(img,coors)
cv.imwrite('crop.jpg',crop)





    












    



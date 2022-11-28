#create and store dataset
import pandas as pd
import numpy as np
from spherical_geometry import polygon
from Utils import *
from nfov import*

HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
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
print(get_one_trace())

def xyz_xy(x,y,z):
    u= x/z
    v=y/z

def get_corner(img):
    trace = get_one_trace()
    trace = [trace[1],trace[2],trace[3]]
    center_point = xyz_xy(trace[1],trace[2],trace[3])
    nfov.toNFOV(img, center_point)

    












    



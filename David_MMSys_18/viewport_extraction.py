#create and store dataset
import pandas as pd
import numpy as np
from spherical_geometry import polygon

HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)



#david.create_and_store_sampled_dataset()
import Read_Dataset as david
#read from dataset
def load_data():
    dataset = david.load_sampled_dataset()
    # df with (dataset, user, video, times, traces)
                # times has only time-stamps
                # traces has only x, y, z (in 3d coordinates)
    print(dataset)
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
def get_xyz():
    ds = load_data()








    



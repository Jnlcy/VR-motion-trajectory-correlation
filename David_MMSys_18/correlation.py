import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

import viewport_extraction as vp

STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'
OUTPUT_FOLDER_FILTERED_FLOW = './David_MMSys_18/FilteredFlows'
VIDEOS = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']


def pearson_global(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarry.flatten(traces)
    r, p = stats.pearsonr(flow,traces)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")   

    return r,p

def pearson_local(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarry.flatten(traces)
    data = [{flow},{traces}]
    df=pd.DataFrame(data,columns=['flow', 'traces'])
    r_window_size = 120
    # Interpolate missing data.
    df_interpolated = df.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated['S1_Joy'].rolling(window=r_window_size, center=True).corr(df_interpolated['S2_Joy'])
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    df.rolling(window=30,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame',ylabel='Pearson r')
    plt.suptitle("Smiling data and rolling window correlation")


def load_filtered_flow(video_name,user):
    user_name = 'david_'+str(user)
    video_folder = os.path.join(OUTPUT_FOLDER_FILTERED_FLOW,video_name)
    path = os.path.join(video_folder, user_name)
    headers = ['Time','x','y','z']
    flow = pd.read_csv(path, header=None)
    flow.columns = headers
    return flow

def data_tidying(video_name,user,flow):
    df = vp.load_data()
    traces = df[video_name][user].loc[:,'traces']
    for i in enumerate(flow.keys()):
        if flow['x']==None:
            traces = traces.drop(labels = i,axis = 0)
    flow = flow.dropna(subset ='x')
    return traces,flow
    

flow =load_filtered_flow('1_PortoRiverside',2)
print(flow)





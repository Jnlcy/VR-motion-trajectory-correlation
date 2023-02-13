import pandas as pd
import numpy as np
from Utils import *
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import random as rd

import viewport_extraction as vp

STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'
OUTPUT_FOLDER_FILTERED_FLOW = './David_MMSys_18/FilteredFlows'
VIDEOS = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']


def pearson_global(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarray.flatten(traces)

    plt.plot(flow,traces,'o')    
        #setting title
    #setting axix label
    plt.xlabel('opticalflow')
    plt.ylabel('traces')
    plt.grid()
    plt.show()
    r, p = stats.pearsonr(flow,traces)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")   

    return r,p


def load_filtered_flow(video_name,user):
    user_name = 'david_'+str(user)
    video_folder = os.path.join(OUTPUT_FOLDER_FILTERED_FLOW,video_name)
    path = os.path.join(video_folder, user_name)
    headers = ['Time','x','y','z']
    flow = pd.read_csv(path, header=None)
    flow.columns = headers
    return flow

def xyz2ab(x,y,z):
    theta,phi = cartesian_to_eulerian(x,y,z)
    theta,phi = eulerian_in_range(theta, phi)
    a=theta/(2*np.pi)
    b=phi/np.pi
    return a,b

def data_tidying(video_name,user):#this function drop all unavailable values in corresponding flow and traces
    user_name = 'david_'+str(user)
    df = vp.load_data()
    #load stored dataframs
    flow = load_filtered_flow(video_name,user)
    #print(flow)
    #find traces
    tmp = df.loc[(df['ds_video'] ==video_name) & (df['ds_user'] ==user_name)]
    traces = tmp['traces']
    traces =traces.to_numpy()
    #store traces in temporaray dataframe
    tmpdf = pd.DataFrame(traces[0],columns = ['Time','x','y','z'])
    #print(tmpdf)
    null_list = flow[flow['x'].isnull()].index.tolist()
    traces = tmpdf.drop(index = null_list)
    flow = flow.dropna(subset ='x')
    
    if len(traces) ==len(flow):
        pass
    else:
        #the last timestamps doesn't exit so drop the last traces
        traces=traces.drop(index = 99)
    
    traces = traces[['x','y','z']].to_numpy()
    flow = flow[['x','y','z']].to_numpy()
    print('Dropped all unavialble cells')
   
    return traces,flow
    

def plot_pearson_global():
    user_values = list(range(0,57))
    for video_name in VIDEOS:
        #print(video_name)
        r_values = []
        for user in user_values:
            #print(user)
            #print(video_name)
            traces,new_flow  = data_tidying(video_name,user)
            r,p = pearson_global(new_flow,traces)
            r_values.append(r)


def plot_ramdom_pearson():
    output_folder = './David_MMSys_18/Pearson'
    # Fixing random state for reproducibility
    np.random.seed(7)
    users = np.arange(57)
    video_sample = rd.choices(VIDEOS,k=)
    user_sample = rd.choices(users, k = 3)
    count =1
    plt.figure(figsize=(10, 6),tight_layout=True) 
    for video_name in video_sample:      
        for user in user_sample:
            #print(user)
            #print(video_name)
            traces,flow  = data_tidying(video_name,user)
            flow = np.ndarray.flatten(flow)
            traces=np.ndarray.flatten(traces)
            r, p = stats.pearsonr(flow,traces)


            plt.subplot(3,3,count)
            plt.plot(flow,traces,'o',ms=2)    
            
            #setting title
            plt.title(video_name+' '+str(user))
            #setting axix label
            plt.xlabel('optical flow',loc='left')
            plt.ylabel('traces')
            plt.grid()
            count+=1
    plt.suptitle('Head Motion against Flow',weight='bold')
    path=os.path.join(output_folder,'Scatter')
    plt.savefig(path)
    plt.show()
    return
            





'''plt.scatter(user_values,r_values,'o')    
        #setting title
        plt.title(video_name)

        #setting axix label
        plt.xlabel('user')
        plt.ylabel('r_value')
        plt.grid()
        plt.show()'''
        


#traces,new_flow = data_tidying('1_PortoRiverside',2)
#r,p = pearson_global(new_flow,traces)
plot_ramdom_pearson()

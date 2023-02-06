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

def pearson_local(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarray.flatten(traces)
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
    for video_name in VIDEOS:
        #print(video_name)
        r_values = []
        user_values = list(range(0,57))
        
        for user in user_values:
            #print(user)
            #print(video_name)
            traces,new_flow  = data_tidying(video_name,user)
            r,p = pearson_global(new_flow,traces)
            r_values.append(r)

            
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
plot_pearson_global()

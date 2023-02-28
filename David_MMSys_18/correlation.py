import pandas as pd
import numpy as np
from Utils import *
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import random as rd
import jenkspy
import plotly.graph_objects as go
import plotly.express as px

import viewport_extraction as vp
from typing import Literal

STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'
OUTPUT_FOLDER_FILTERED_FLOW = './David_MMSys_18/FilteredFlows'
VIDEOS = ['1_PortoRiverside', '2_Diner',  '4_Ocean', '5_Waterpark', '6_DroneFlight',
            '8_Sofa', '9_MattSwift', '10_Cows', '12_TeatroRegioTorino', '13_Fountain', '14_Warship',
            '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'hight': 'red'}


def pearson_global(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarray.flatten(traces)

   
    r, p = stats.pearsonr(flow,traces)
    #print(f"Scipy computed Pearson r: {r} and p-value: {p}")   

    return r,p

def user_attraction(flows,traces):
    user_attraction = np.zeros(len(flows[:]))
    for i in range(len(flows[:])-1):
        userAtr =  np.dot(flows[i],traces[i+1]-traces[i])
        user_attraction[i] = userAtr
    return np.mean(user_attraction)



def load_filtered_flow(video_name,user):
    user_name = 'david_'+str(user)
    video_folder = os.path.join(OUTPUT_FOLDER_FILTERED_FLOW,video_name)
    path = os.path.join(video_folder, user_name)
    headers = ['Time','x_e','y_e','z_e','x_f','y_f','z_f']
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
    null_list = flow[flow['x_e'].isnull()].index.tolist()
    traces = tmpdf.drop(index = null_list)
    flow = flow.dropna(subset ='x_e')
    
    if len(traces) ==len(flow):
        pass
    else:
        #the last timestamps doesn't exit so drop the last traces
        traces=traces.drop(index = 99)
    
    time = traces[['Time']].to_numpy()
    traces = traces[['x','y','z']].to_numpy()
    endpoint = flow[['x_e','y_e','z_e']].to_numpy()
    flow_vector = flow[['x_f','y_f','z_f']].to_numpy()
    #print('Dropped all unavialble cells')
   
    return time, traces,endpoint,flow_vector   


def cal_pearson(video_name):
    users = np.arange(57)
    user_cor  = []
    for i in range(len(users)):
        time,traces,endpoint,flow_vector = data_tidying(video_name,users[i])
        cor,p = pearson_global(flow_vector,traces)
        print(str(users[i])+" 's overall correlation with movement is "+ str(cor))
        user_cor.append(cor)
    
    user_min=user_cor.index(min(user_cor))
    user_max=user_cor.index(max(user_cor))

    text = 'Highest correlation: user '+str(user_max)+'\nLowest correlation: user '+str(user_min)
    print(text)

    return user_cor




def get_class_thresholds(data) :
  _, threshold_medium, threshold_hight, _ = jenkspy.jenks_breaks(data, n_classes=3)
  return threshold_medium, threshold_hight

def get_class_name(x: float, threshold_medium: float,
                   threshold_hight: float) -> Literal['low', 'medium', 'hight']:
  return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')

def calc_attraction(video_name):
    users = np.arange(57)
    user_atr  = []
    for i in range(len(users)):
        time,traces,endpoint,flow_vector = data_tidying(video_name,users[i])
    
        atr = user_attraction(flow_vector,traces)
        print(str(users[i])+" 's overall attraction to movement is "+ str(atr))
        user_atr.append(atr) 

    #calculate min and max attraction
    user_min=user_atr.index(min(user_atr))
    user_max=user_atr.index(max(user_atr))

    text = 'Highest attraction: user '+str(user_max)+'\nLowest attrction: user '+str(user_min)
    print(text)

    return user_atr

def plotH_attraction(video_name):

    
    atr= np.array(calc_attraction(video_name))
    threshold_medium, threshold_hight = get_class_thresholds(atr)
    atr1 =atr [atr<threshold_medium]
    atr2 =atr[(atr<threshold_hight)  & (atr>=threshold_medium)]
    atr3 = atr[atr>=threshold_hight]
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=atr1,name = 'low attraction'))
    fig.add_trace(go.Histogram(x=atr2,name = 'medium attraction'))
    fig.add_trace(go.Histogram(x=atr3,name = 'high attraction'))


    fig.update_layout(
        title_text=video_name, # title of plot
        xaxis_title_text='Attraction', # xaxis label
        yaxis_title_text='Count', # yaxis label
        barmode = 'stack'
    
        ) # gap between bars of the same location coordinates)


    fig.show()
    #plot histgram

    return
    
def plotH_pearson(video_name):
    cor= np.array(cal_pearson(video_name))
    threshold_medium, threshold_hight = get_class_thresholds(cor)
    atr1 =cor [cor<threshold_medium]
    atr2 =cor[(cor<threshold_hight)  & (cor>=threshold_medium)]
    atr3 = cor[cor>=threshold_hight]
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=atr1,name = 'low correlation'))
    fig.add_trace(go.Histogram(x=atr2,name = 'medium correlation'))
    fig.add_trace(go.Histogram(x=atr3,name = 'high correlation'))


    fig.update_layout(
        title_text=video_name, # title of plot
        xaxis_title_text='Attraction', # xaxis label
        yaxis_title_text='Count', # yaxis label
        barmode = 'stack'
    
        ) # gap between bars of the same location coordinates)


    fig.show()
    #plot histgram

    return

def trace_flow_comparison(video_name,high,low):
    #create figure
    mpl.style.use('default')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 4),tight_layout=True)    

    #lowest attracttion
    time,traces,endpoint,flow_vector = data_tidying(video_name,low)
    flow_x,flow_y,flow_z = flow_vector[:,0],flow_vector[:,1],flow_vector[:,2]
    trace_x,trace_y,trace_z = traces[:,0],traces[:,1],traces[:,2]

    theta_t,phi_t,theta_f, phi_f= np.zeros_like(flow_x),np.zeros_like(flow_x),np.zeros_like(flow_x),np.zeros_like(flow_x)
    for i in range(len(flow_x)):
        theta_t[i] ,phi_t[i] = cartesian_to_eulerian(trace_x[i],trace_y[i],trace_z[i]) 
        theta_t[i] ,phi_t[i] = eulerian_in_range(theta_t[i] ,phi_t[i])
        theta_f[i], phi_f[i]= cartesian_to_eulerian(flow_x[i],flow_y[i],flow_z[i]) 
        

    ax1.quiver(theta_t[:-1] ,phi_t[:-1],theta_t[1:]-theta_t[:-1],phi_t[1:]-phi_t[:-1]
    ,color= 'C2', scale_units='xy', angles = 'xy',scale=1,  label ='User head position')
    ax1.quiver(theta_t ,phi_t,theta_f,phi_f,color ='C0',width = 0.005,label = 'Main optical flow')
    ax1.set_xlabel('theta')
    ax1.set_ylabel('phi')
    
    ax1.set_title('Low Attraction: User '+str(low))
    ax1.legend()
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-np.pi/2.0,np.pi/2.0])


    time,traces,endpoint,flow_vector = data_tidying(video_name,high)
    flow_x,flow_y,flow_z = flow_vector[:,0],flow_vector[:,1],flow_vector[:,2]
    trace_x,trace_y,trace_z = traces[:,0],traces[:,1],traces[:,2]

    theta_t,phi_t,theta_f, phi_f= np.zeros_like(flow_x),np.zeros_like(flow_x),np.zeros_like(flow_x),np.zeros_like(flow_x)
    for i in range(len(flow_x)):
        theta_t[i] ,phi_t[i] = cartesian_to_eulerian(trace_x[i],trace_y[i],trace_z[i]) 
        theta_t[i] ,phi_t[i] = eulerian_in_range(theta_t[i] ,phi_t[i])
        theta_f[i], phi_f[i]= cartesian_to_eulerian(flow_x[i],flow_y[i],flow_z[i]) 
        



    ax2.quiver(theta_t[:-1] ,phi_t[:-1],theta_t[1:]-theta_t[:-1],phi_t[1:]-phi_t[:-1],color = 'C2',
    scale_units='xy', angles = 'xy',scale=1, label ='User head position')
    ax2.quiver(theta_t ,phi_t,theta_f,phi_f,color='C0',width = 0.005, label = 'Main optical flow')
    ax2.set_xlabel('theta')
    ax2.set_ylabel('phi')
    ax2.set_title('High Attraction: User '+str(high))
    ax2.legend()
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi/2.0,np.pi/2.0])

    plt.show()

    return
#Cluster video montions into High/medium/low and plot
def motion_analysis():
    df  = vp.load_flow()
    print(df)
    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows',  '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
    video_averages=[]
    for video in videos:  
        sample = df.query(f"video=='{video}'")
        averages=[]
        for t in sample['time'].values:
            
            flow_sample= sample.query(f"time=='{t}'")  
            flows = flow_sample['optical flow']

            flow_magnitudes = []
            for flow in flows:
                magnitude = (flow[2]-flow[0])*(flow[2]-flow[0])+(flow[3]-flow[1])*(flow[3]-flow[1])
                flow_magnitudes.append(magnitude)
            average = round(np.average(flow_magnitudes),2)
            averages.append(average)
        video_average = round(np.average(averages),2)    
        video_averages.append(video_average)
    
    df_vm = pd.DataFrame(list(zip(videos,video_averages)),columns=['video', 'average flow'])
    df_vm['Value Group']=''
    threshold_medium, threshold_hight = get_class_thresholds(video_averages)
    print(df_vm)
    for ind in df_vm.index:
        value = df_vm.loc[ind]['average flow'] 
        if value <=threshold_medium:
            df_vm.at[ind,'Value Group'] = 'Low Motion'
        elif value >= threshold_hight:
            df_vm.at[ind,'Value Group'] = 'High Motion'
        else:
            df_vm.at[ind,'Value Group'] = 'Medium Motion'
    df_vm.to_csv('motion analysis.csv')
  
    return df_vm
#test mean magnitude

#print(motion_analysis())
     

                                                                                   

#motion analysis:
'''
#test High motion: video  4_Ocean:
plotH_attraction('4_Ocean')
trace_flow_comparison('4_Ocean',6,3)

#test Medium motion: video 10_Cows: 
plotH_attraction('10_Cows')
trace_flow_comparison('10_Cows',5,15)

#test low motion: 16_Turtle
plotH_attraction('16_Turtle')
trace_flow_comparison('16_Turtle',47,2)
'''

#user analysis:
#test High motion: video  4_Ocean:
#plotH_pearson('4_Ocean')
#trace_flow_comparison('4_Ocean',55,15)

#test Medium motion: video 10_Cows: 
#plotH_pearson('10_Cows')
#trace_flow_comparison('10_Cows',15,26)

#test low motion: 16_Turtle
plotH_attraction('16_Turtle')
trace_flow_comparison('16_Turtle',47,2)





    


    





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
#plot_random_pearson()

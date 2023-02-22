import pandas as pd
import numpy as np
from Utils import *
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import random as rd

import viewport_extraction as vp
from mpl_toolkits import mplot3d

STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'
OUTPUT_FOLDER_FILTERED_FLOW = './David_MMSys_18/FilteredFlows'
VIDEOS = ['1_PortoRiverside', '2_Diner',  '4_Ocean', '5_Waterpark', '6_DroneFlight',
            '8_Sofa', '9_MattSwift', '10_Cows', '12_TeatroRegioTorino', '13_Fountain', '14_Warship',
            '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']


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

def user_attraction(flows,traces):
    user_attraction = np.zeros(len(flows[:]))
    for i in range(len(flows[:])-1):
        userAtr =  np.dot(flows[i+1],traces[i+1]-traces[i])
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


def plot_random_pearson():
    output_folder = './David_MMSys_18/Pearson'
    # Fixing random state for reproducibility
    np.random.seed(7)
    users = np.arange(57)
    video_sample = rd.choice(VIDEOS)
    user_sample = rd.choice(users)

    
    #print(user)
    #print(video_name)
    traces,flow,vector  = data_tidying(video_sample,user_sample)
    
    t,flow_x,flow_y,flow_z = flow[:,0],flow[:,1],flow[:,2],flow[:,3]
    trace_x,trace_y,trace_z = traces[:,1],traces[:,2],traces[:,3]
    

    #create a figure
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    title = 'User '+str(user_sample)+' watching '+video_sample
    theta_t,phi_t,theta_f, phi_f= np.zeros_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)
    for i in range(len(t)):
        theta_t[i] ,phi_t[i] = cartesian_to_eulerian(trace_x[i],trace_y[i],trace_z[i]) 
        theta_f[i], phi_f[i]= cartesian_to_eulerian(flow_x[i],flow_y[i],flow_z[i]) 

    ax.plot3D(t,theta_t ,phi_t,'grey',label ='User head position')  
    ax.scatter3D(t,theta_f, phi_f,label = 'Mean optical flow end points')
    
    ax.set_xlabel('time(second)')
    ax.set_ylabel('theta')
    ax.set_zlabel('phi')
    ax.set_title(title)
    ax.legend()

            
    path=os.path.join(output_folder,title)
    fig.savefig(path)
    fig.show()
    return
            
def plotH_attraction(video_name):
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
    #plot histgram

    f1,ax = plt.subplots()
    ax.hist(user_atr,bins = 5, edgecolor="white")
    ax.set_title("User's attraction distribution for "+video_name)
    ax.set_xlabel('User Attraction to Movement')
    ax.set_ylabel('Count')
    text = 'Highest attraction: user '+str(user_max)+'\nLowest attrction: user '+str(user_min)
    print(text)
    plt.show()
    

    #spacial visualise correlataion
    #f2 = trace_flow_comparison(video_name,user_min,user_max)
    #f2.show()


    return

def trace_flow_comparison(video_name,user_min,user_max):
    #create figure
    mpl.style.use('default')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 4),tight_layout=True)    

    #lowest attracttion
    time,traces,endpoint,flow_vector = data_tidying(video_name,user_min)
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
    
    ax1.set_title('High Attraction: User '+str(user_max))
    ax1.legend()
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-np.pi/2.0,np.pi/2.0])


    time,traces,endpoint,flow_vector = data_tidying(video_name,user_max)
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
    ax2.set_title('Low Attraction: User '+str(user_min))
    ax2.legend()
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi/2.0,np.pi/2.0])

    plt.show()

    return


trace_flow_comparison('17_UnderwaterPark',2,34)




    


    
#test plotH
#plotH_attraction('17_UnderwaterPark')




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

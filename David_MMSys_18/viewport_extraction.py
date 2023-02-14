
#create and store dataset
import pandas as pd
import numpy as np
from ast import literal_eval
import cv2 as cv
import os
import argparse
from Utils import *

HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
HEIGHT=1920
WIDTH=3840
STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'
OUTPUT_FOLDER_FILTERED_FLOW = './David_MMSys_18/FilteredFlows'
VIDEOS = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows',  '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']

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
             video,
            dataset[user][video]
            ) for user in dataset.keys() for video in dataset[user].keys()]

    tmpdf = pd.DataFrame(data, columns=[
                    'ds', 'ds_user', 'ds_video',
                    'traces'])
    return tmpdf
#get xyz corner of the view port


def get_traces(ds,idx):
    traces = ds.loc[idx]['traces']
    video = ds.loc[idx]['ds_video']
    return traces, video




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
        #print(points)
        
            
        _fov_points[(trace[1], trace[2], trace[3])] = points
    return _fov_points[(trace[1], trace[2], trace[3])]



def compare_lucas_kanade_method(video_path,t):
    MILLISECONDS = 1000
    cap = cv.VideoCapture(video_path)
    # params for ShiTomasi corner detection
 
    feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=7, blockSize=1)
    # Parameters for lucas kanade optical flow

    lk_params = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10, 0.03),
    )
     
   
    #create old frame
    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS))
    ret, old_frame = cap.read()
    if not ret:
        return None,None
        
    #convert the frame into grey scale
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)



    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS+200))
    ret, frame = cap.read()
    if not ret:
        return None, None

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #save frame
    img = cv.addWeighted(old_frame, 0.5,frame,0.5,0.0)
    #cv.imshow("image.jpg",img)
        
    # forward-backwoard error detection
    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )
        

        # If no flow, look for new points
    if p1 is None:

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Get rid of old lines
        mask = np.zeros_like(old_frame)
        #cv.imshow ('frame', frame)

    else:
        # Select good points
            
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        #filter the useful optical flow
        flow,mask,frame=plotflow(mask,frame,good_new,good_old)
        flow_image = cv.add(img,mask)
        cv.imshow('flow wih image', flow_image)
        cv.waitKey(25) & 0xFF
    
        
    cap.release()
    return good_old, good_new

                 
    #save_optical_flow(flow)
    
def ab2xyz(a,b): #project 2d piont onto 2d unit square

    theta = a/WIDTH *2* np.pi # The longitude ranges from 0, to 2*pi
    phi = b/HEIGHT* np.pi # The latitude ranges from 0 to pi, origin of equirectangular in the top-left corner
    xyz = eulerian_to_cartesian(theta,phi)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    return x,y,z


def plotflow(mask,frame,new,old):
    flow = []
    for i, (new, old) in enumerate(zip(new, old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Green color in BGR
        color = (0, 255, 0)      
        flow.append([a,b,c,d])
        mask = cv.arrowedLine(mask, (int(c),int(d)), (int(a),int (b)), color,2)
        frame = cv.circle(frame, (int(a),int(b) ), 5, color, -1)
   
    
    return flow,mask,frame

def flow_filter(new,old,corners):
    flow_vector = []
    filtered_new = []
    

    for i, (new, old) in enumerate(zip(new, old)):
        
        a, b = new.ravel()
        c, d = old.ravel()
        
        #project 2d points onto 3d sphere
        x_n,y_n,z_n = ab2xyz(a,b)
        #print([x_n,y_n,z_n])
        x_o,y_o,z_o =ab2xyz(c,d)
        

        x_lim_up = max(corners[:,0])
        x_lim_down = min(corners[:,0])
        y_lim_up = max(corners[:,1])
        y_lim_down = min(corners[:,1])
        z_lim_up = max(corners[:,2])
        z_lim_down = min(corners[:,2])
        
        
        #print([x_lim_up,x_lim_down, y_lim_up,y_lim_down,z_lim_up,z_lim_down])
         #and (y_lim_down<y_n<y_lim_up)
        if (x_lim_down<=x_n<=x_lim_up) and (y_lim_down<=y_n<=y_lim_up) and(z_lim_down<=z_n<=z_lim_up):

        
            filtered_new.append([x_n,y_n,z_n])

            x_v,y_v,z_v = x_n-x_o,y_n-y_o,z_n-z_o
            flow_vector.append([x_v,y_v,z_v])


        else:
            pass
        #print(filtered_old)
    flow_vector = np.array(flow_vector)
    filtered_new = np.array(filtered_new)
    return flow_vector,filtered_new

def save_flow(): #calculate optical flow of the video
    for video in VIDEOS:
        #find the path of the video
        video_path = os.path.join(STIMULI_FOLDER,video+".mp4")
        
        print(video)
        #compute optical flow of the entire video
        video_flow = []
        timestamps= np.linspace(0,19.8,100)

        for t in timestamps:
            good_old, good_new = compare_lucas_kanade_method(video_path,t)
            if good_old is not None:
                df = pd.DataFrame(zip(good_old,good_new),columns = ['good_old','good_new'])
                df2 = pd.DataFrame(df.good_old.to_list(),columns = ['old_x','old_y'])
                df2[['new_x','new_y']] = pd.DataFrame(df.good_new.to_list(),index= df2.index)

                video_folder = os.path.join(OUTPUT_FOLDER_FLOW,video)
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                path = os.path.join(video_folder,"{:.1f}".format(t))
                df2.to_csv(path,index=False)
            print('flow at ',"{:.1f}".format(t), ' saved')
        print(video+'optical flow saved ')
    print('All optical flow saved')

#save_flow()    
#test the saving flow function

def load_flow(t,video):
    video_folder = os.path.join(OUTPUT_FOLDER_FLOW,video)
    path = os.path.join(video_folder,"{:.1f}".format(t))
    df = pd.read_csv(path,index_col=False )
    if df is None:
        path = os.path.join(video_folder,"{:.1f}".format(t-0.2))
    old = df[['old_x','old_y']].to_numpy()
    new = df[['new_x','new_y']].to_numpy()

    return old,new

#test load_flow function    
#old,new =load_flow(1.0,'18_Bar')
#print(new)

def nanmean(a):
    if a.size == 0:
        return np.NaN
    else:
        return np.nanmean(a,axis = 0)


def store_filtered(df,video_name,user):#save filtered flow
    video_folder = os.path.join(OUTPUT_FOLDER_FILTERED_FLOW,video_name)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    path = os.path.join(video_folder, user)
    df.to_csv(path, header=False, index=False)
    

def save_filteredFlow():#add optical flow for each user to the dataframe

    ds = load_data()
    #print(ds)
    ds2 = ds.assign(Mean_flow =None)
   
   
    
    for i in range(len(ds2)):
        
        traces,video_name = get_traces(ds,i)
        user = ds2.loc[i]['ds_user']
        corners_video = []
        endpoint_video = []
        flow_video=[]
        #iterate through each time stamps and calculate flow 
        for j in range(len(traces)-1):
            corners = fov_points(traces[j])   
            t = traces[j][0]
            corners_video.append([corners])
            old,new =load_flow(t,video_name)
            
            flow_vector,new_filtered = flow_filter(old,new,corners)
            
           
            mean_new = nanmean(new_filtered)
            mean_flow = nanmean(flow_vector)
            if mean_new is not np.NaN:
                x,y,z = mean_new.ravel()
                x_f,y_f,z_f = mean_flow .ravel()
                endpoint_video.append([round(t,1),x,y,z])
                flow_video.append([x,y,z])
            else:
                endpoint_video.append([round(t,1),None,None,None])
                flow_video.append([None,None,None])


            
        df1 = pd.DataFrame(endpoint_video)
        df2 = pd.DataFrame(flow_video)
        df = pd.concat([df1,df2.reindex(df1.index)], axis=1)
        print(df)

        store_filtered(df,video_name,user)
         
        #add flow and traces into the dataset
        
        ds2['Mean_flow'][i]=flow_video
        print('Finish filtering '+user+','+video_name)
    
    return(ds2)


#save_filteredFlow()#test add flow
#ds2.to_csv('Optical Flow')#save the new df 

#print(ds2)


            
            

#test dataset assign
"""ds = load_data()
traces,video_name = get_traces(ds,0)
print(video_name)
video_path = os.path.join(STIMULI_FOLDER,video_name +'.mp4')
print(video_path)
corners = fov_points(traces[10])
print(corners)
t = traces[10][0]
print(t)
filtered_flow = compare_lucas_kanade_method(video_path,2,corners)
print(filtered_flow)"""


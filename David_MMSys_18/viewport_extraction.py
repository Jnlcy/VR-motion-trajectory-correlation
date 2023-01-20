
#create and store dataset
import pandas as pd
import numpy as np
import cv2 as cv
import os
from spherical_geometry import polygon
from Utils import *
from nfov import *

HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
HEIGHT=1920
WIDTH=3840
STIMULI_FOLDER = './David_MMSys_18/Stimuli'
OUTPUT_FOLDER_FLOW ='./David_MMSys_18/Flows'

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



def compare_lucas_kanade_method(video_path,t,corners):
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
   
    
    #fps = cap.get(cv.CAP_PROP_FPS)


    #create old frame
    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS))
    ret, old_frame = cap.read()
    
        
    #convert the frame into grey scale
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)



    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS+200))
    ret, frame = cap.read()

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
        filtered_flow,mask,frame=flow_filter(mask,frame,good_new,good_old,corners)
        #flow_image = cv.add(img,mask)
        #cv.imshow('flow wih image', flow_image)
    
        
    cap.release()
    return filtered_flow

                 
    #save_optical_flow(flow)
    
def ab2xyz(a,b): #project 2d piont onto 2d unit square

    theta = a/WIDTH *2* np.pi # The longitude ranges from 0, to 2*pi
    phi = b/HEIGHT* np.pi # The latitude ranges from 0 to pi, origin of equirectangular in the top-left corner
    xyz = eulerian_to_cartesian(theta,phi)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    return x,y,z



def flow_filter(mask,frame,new,old,corners):
    filtered_flow=[]
    color = (0, 255, 0)  

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
                
            
            # Green color in BGR
            mask = cv.arrowedLine(mask, (int(c),int(d)), (int(a),int (b)), color,2)
            frame = cv.circle(frame, (int(a),int(b) ), 5, color, -1)   
            
            filtered_flow.append([x_o,y_o,z_o,x_n,y_n,z_n]) 

            #print([a,b,c,d])
                 
            #mask = cv.arrowedLine(mask, (int(c),int(d)), (int(a),int(b)),color,1)
            #print([a,b,c,d])
            #frame = cv.circle(frame, (int(a),int(b)), 5, color, -1)
                
        else:
            pass

    img= cv.add(frame, mask)
    cv.imshow("frame.jpg", img)
    cv.waitKey(25) & 0xFF
        
    flow = np.array(filtered_flow)
    print(flow)
    return flow,mask,frame

def add_flow():

    ds = load_data()
    #print(ds)
    ds2 = ds.assign(Corners = None,Optical_flow =None)
   
    
    for i in range(len(ds2)):
        #print(i)
        traces,video_name = get_traces(ds,i)
        video_path = os.path.join(STIMULI_FOLDER,video_name+'.mp4')
        corners_video = []
        flow_video = []
        #iterate through each time stamps and calculate flow 
        for j in range(len(traces)-1):
            corners = fov_points(traces[j])   
            print(corners)
            t = traces[j][0]
            corners_video.append([t,corners])
            flow = compare_lucas_kanade_method(video_path,t,corners)
            flow_video.append([flow])
        #add flow and traces into the dataset
        ds2['Corners'][i] = corners_video
        ds2['Optical_flow'][i]=flow_video
    
    return(ds2)






        

ds2 = add_flow()
ds2.to_csv('Optical Flow',header =['traces','Corners','Optical_flow'])


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








            








    












    


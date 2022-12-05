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

'''def get_one_trace() -> np.array:
    dataset = load_data()
    return dataset.iloc[0]['traces'][0]

trace = get_one_trace()
print(trace)'''


def get_traces(ds,idx):
    traces = ds.loc[idx]['traces']
    video = ds.loc[idx]['ds_video']
    return traces, video




    


def xyz2uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)
    coor_x = (u / (2 * np.pi) + 0.5) * WIDTH- 0.5
    coor_y = (-v / np.pi + 0.5) * HEIGHT - 0.5
    return np.concatenate([coor_x, coor_y], axis=-1)



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
        print(points)
        
        #convert into 2d corners
        points_converted = np.array([
                xyz2uv(points[0]),
                xyz2uv(points[1]),
                xyz2uv(points[2]),
                xyz2uv(points[3])
            ])
            
        _fov_points[(trace[1], trace[2], trace[3])] = points_converted
    return _fov_points[(trace[1], trace[2], trace[3])]



def crop_image(img,corners):
    
    x = []
    y = []
    for view_corner in corners:
        xx,yy = (np.split(view_corner,2,axis=-1))
        x.append(xx)
        y.append(yy)
    
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    crop_img = img[int(y_min):int(y_max),int(x_min):int(x_max)]
    # cv.imshow('cropped',crop_img)
    return crop_img


def compare_lucas_kanade_method(video_path,t,corners):
    cap = cv.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    
    
    feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=7, blockSize=1)
    # Parameters for lucas kanade optical flow

    lk_params = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10, 0.03),
    )
   
    # Take first frame and find corners in it
    
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for plotting purposes
    mask = np.zeros_like(old_frame)
    

    
    
    MILLISECONDS = 1000
    #fps = cap.get(cv.CAP_PROP_FPS)
    print(t)

    #create old frame
    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS))
    ret, old_frame = cap.read()
    old_frame = crop_image(old_frame,corners)
        
     #convert the frame into grey scale
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)


    cap.set(cv.CAP_PROP_POS_MSEC, float(t*MILLISECONDS+200))
    ret, frame = cap.read()
    frame = crop_image(frame,corners)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #save frame
    img = cv.addWeighted(old_frame, 0.5,frame,0.5,0.0)
    frame_name=os.path.join('images','frame at '+str(t)+' th second.jpg')
    #cv.imwrite(frame_name,img)
        

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
        cv.imshow ('frame', frame)

    else:
        # Select good points
            
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        #plot optical flow
        flow,frame=plotflow(mask,frame,good_new,good_old)
        #img=plotquiver(t,frame,good_new,good_old)
            
        
        k = cv.waitKey(25) & 0xFF
        flow_image = cv.add(img,mask)
        cap.release()
    
        #flow_name =  os.path.join('images','flow'+str(t)+".jpg")
        #flow_image_name  =  os.path.join('images',"flow with image "+str(t)+".jpg")      
        #cv.imwrite(flow_name,flow)
        cv.imshow('flow with image', flow_image)
         
    
    return flow

                 
    #save_optical_flow(flow)
    
    

def plotflow(mask,frame,new,old):
    for i, (new, old) in enumerate(zip(new, old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Green color in BGR
        color = (0, 255, 0)      
        
        mask = cv.arrowedLine(mask, (int(c),int(d)), (int(a),int (b)), color,2)
        frame = cv.circle(frame, (int(a),int(b) ), 5, color, -1)
   
    
    return mask,frame


def add_flow():
    ds = load_data()
    print(ds)
    ds2 = ds.assign(Corners = None,Optical_flow =None)
    ds2 = ds2.head(2)
    for i in range(len(ds2)):
        print(i)
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
            flow_video.append([t,flow])
        #add flow and traces into the dataset
        ds2['Corners'][i] = corners_video
        ds2['Optical_flow'][i]=flow_video
    
    return(ds2)

ds2 = add_flow()
#print(ds2)
            
            


#test dataset assign
'''ds = load_data()
traces,video_name = get_traces(ds,0)
print(video_name)
video_path = os.path.join(STIMULI_FOLDER,video_name +'.mp4')
print(video_path)
corners = fov_points(traces[9])
print(corners)
t = traces[9][0]
print(t)
flow = compare_lucas_kanade_method(video_path,1.8,corners)
print(flow)'''




#ds2 = ds.assign(Corners = None,Optical_flow =None)
# print(ds2)
            








    












    



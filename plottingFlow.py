from optparse import make_option
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def plotflow(mask,frame,new,old):
    for i, (new, old) in enumerate(zip(new, old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Green color in BGR
        color = (0, 255, 0)      
        
        mask = cv2.arrowedLine(mask, (int(c),int(d)), (int(a),int (b)), color,2)
        frame = cv2.circle(frame, (int(a),int(b) ), 5, color, -1)
   
    
    return mask,frame



'''

def plotquiver(t,frame,new,old):
    
    x=[]
    y=[]
    dx=[]
    dy=[]

    for i, (new, old) in enumerate(zip(new, old)):
        a, b = new.ravel()
        c, d = old.ravel()
        x.append(c)
        y.append(d)
        dx.append(a-c)
        dy.append(b-d)

   

    #plot 
    
    plt.imshow(frame)
    plt.quiver(x,y,dx,dy,angles='xy', scale_units='xy', scale=0.5, pivot='mid',color='y')
    plt.savefig("flow"+str(t)+".jpg")

'''  

def compare_lucas_kanade_method(video_path):
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    
    
    
    feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=7, blockSize=1)
    # Parameters for lucas kanade optical flow

    lk_params = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03),
    )
   
    # Take first frame and find corners in it
    
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for plotting purposes
    mask = np.zeros_like(old_frame)
    

    
    
    for t in [1,4,14]:
        MILLISECONDS = 1000
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt =1/fps
        print(t)

        #create old frame
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t*MILLISECONDS))
        ret, old_frame = cap.read()
        
        #convert the frame into grey scale
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)


        cap.set(cv2.CAP_PROP_POS_MSEC, float(t*MILLISECONDS+200))
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #save frame
        img = cv2.addWeighted(old_frame, 0.5,frame,0.5,0.0)
        frame_name=os.path.join('images','frame at '+str(t)+' th second.jpg')
        cv2.imwrite(frame_name,img)
        



        if not ret:
            break
        
        # forward-backwoard error detection
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        

        # If no flow, look for new points
        if p1 is None:

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            # Get rid of old lines
            mask = np.zeros_like(old_frame)
            cv2.imshow ('frame', frame)

        else:
        # Select good points
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        
            #plot optical flow
            flow,frame=plotflow(mask,frame,good_new,good_old)
            #img=plotquiver(t,frame,good_new,good_old)
            
        
            k = cv2.waitKey(25) & 0xFF
            flow_image = cv2.add(img,mask)

            flow_name =  os.path.join('images','flow'+str(t)+".jpg")
            flow_image_name  =  os.path.join('images',"flow with image "+str(t)+".jpg")      
            cv2.imwrite(flow_name,flow)
            cv2.imwrite(flow_image_name, flow_image) 
            
      

    #save_optical_flow(flow)
    
    cap.release()


def main():
    STIMULI_FOLDER = './David_MMSys_18/Stimuli'
    path = os.path.join(STIMULI_FOLDER,'4_Ocean.mp4')
    
    
    
    flow=compare_lucas_kanade_method(path)
    
    

    

if __name__ == "__main__":
    main()
from optparse import make_option
import cv2
import numpy as np
import os


#def read_fixation():

STIMULI_FOLDER = './David_MMSys_18/Stimuli'
def save_optical_flow(flow):
    
    np.savetxt('flow.csv',flow,delimiter=",")



def lucas_kanade_method(video_path):
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    cap.set(cv2.CAP_PROP_FPS, 200)



    feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=7, blockSize=1)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    optical_flow=[]

    idx = 0
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )


        # If no flow, look for new points
        if p1 is None:

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # Get rid of old lines
            mask = np.zeros_like(old_frame)

            cv2.imshow('frame', frame)
        else:
        # Select good points
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                color = (0, 255, 0) 
                optical_flow.append([int(idx),time,a,b])
                mask = cv2.arrowedLine(mask, (int(a),int (b)), (int(c),int(d)), color, 2)
                frame = cv2.circle(frame, (int(a),int(b) ), 5, color, -1)
            img = cv2.add(frame, mask)
            cv2.imshow("frame.jpg", img)
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break
            if k == ord("c"):
                mask = np.zeros_like(old_frame)
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        idx +=1    
    cap.release()
    return optical_flow
    #save_optical_flow(flow)

    


#def viewport_filer()
def main():

    path = os.path.join(STIMULI_FOLDER,'4_Ocean.mp4')

    flow=lucas_kanade_method(path)
    print(flow)
    save_optical_flow(flow)




if __name__ == "__main__":
    main()
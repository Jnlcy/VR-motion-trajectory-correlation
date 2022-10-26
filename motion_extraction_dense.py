import cv2
import numpy as np


import matplotlib.pyplot as plt


def save2File(flow,file_path):
    
    np.savetxt(file_path,flow,delimiter=",")

def plotMag(mag):
    motion = np.mean(mag, axis=1)
    time = list(range(len(motion)))
    plt.axis([0,360,0,2])
    plt.xlabel('The nth frame')
    plt.ylabel('Average Flow Magnitude')
    plt.plot(time,motion)
    plt.title('Change of Average Optical Flow Magnitude over Frames')
    plt.savefig('magnitudePlot.png')
    plt.show()
        
#def plotFlow(flow)


def dense_flow(video_path):
#grab video from source
    cap = cv2.VideoCapture(video_path)
    fps= cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = round(frames / fps)


    #grab first frame
    ret, frame1 = cap.read()

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    dense_flow_list = []
    magnitude = []
    while True:
        #get next frame
        ret, frame2 = cap.read()
        if not ret:
            print(flow)
            break
        else:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            
            #visualise
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            magnitude.append(mag)
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2',rgb)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # Frame are read by intervals of 10 millisecond. The programs breaks out of the while loop when the user presses the ‘s’ key
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',rgb)
        # Now update the previous frame and previous points
            prvs = next
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    #accumulate all optical flow
    
    cap.release()
    cv2.destroyAllWindows()
    
    return mag,fps,seconds

def main():
    video_path = ('people.mp4')
    mag,frames,fps = dense_flow(video_path)
    print (len(mag))
    save2File(mag,'magnitude.csv')
    plotMag(mag)




if __name__ == "__main__":
    main()
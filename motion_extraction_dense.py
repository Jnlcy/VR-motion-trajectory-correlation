import cv2
import numpy as np
def dense_flow(video_path):
#grab video from source
    cap = cv2.VideoCapture(video_path)

    #grab first frame
    ret, frame1 = cap.read()
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    while True:
        #get next frame
        ret, frame2 = cap.read()
        if not ret:

            break
        else:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #visualise
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
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
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = ('people.mp4')
    dense_flow(video_path)


if __name__ == "__main__":
    main()
from django.shortcuts import render,HttpResponse # render & HTTP response are functions used for http response
import requests # this line will imort request module which are used to generate HTTP request to other website

def button(request):
# button is a view function which takes request as input. 
# IN Django a view function is used for processing an incoming http request and returning a Http response

    return render (request,'index.html')
def canvas(request):
    # All the imports go here
    import cv2
    import numpy as np
    import mediapipe as mp
    from collections import deque


    # Giving different arrays to handle colour points of different colour
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]


    # These indexes will be used to mark the points in particular arrays of specific colour
    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    #The kernel to be used for dilation purpose 
    kernel = np.ones((5,5),np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((471,636,3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils


    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        # Read each frame from the webcam
        ret, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
        frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
        frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
        frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
        frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # # print(id, lm)
                    # print(lm.x)
                    # print(lm.y)
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)

                    landmarks.append([lmx, lmy])


                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0],landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0],landmarks[4][1])
            cv2.circle(frame, center, 3, (0,255,0),-1)
            print(center[1]-thumb[1])
            if (thumb[1]-center[1]<30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140: # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paintWindow[67:,:,:] = 255
                elif 160 <= center[0] <= 255:
                        colorIndex = 0 # Blue
                elif 275 <= center[0] <= 370:
                        colorIndex = 1 # Green
                elif 390 <= center[0] <= 485:
                        colorIndex = 2 # Red
                elif 505 <= center[0] <= 600:
                        colorIndex = 3 # Yellow
            else :
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
        # Append the next deques when nothing is detected to avois messing up
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw lines of all the colors on the canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        # for j in range(len(points[0])):
        #         for k in range(1, len(points[0][j])):
        #             if points[0][j][k - 1] is None or points[0][j][k] is None:
        #                 continue
        #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Output", frame) 
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cv2.destroyAllWindows()
    cap.release()


####----------------------------------------------------Virtual Canvas Code ends here--------------------------------------













def mouse(request):
    # Imports

    import cv2
    import mediapipe as mp
    import pyautogui # it is for controlling the mouse and keyboard
    import math
    from enum import IntEnum
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    # Pycaw is used for controlling the volume
    from google.protobuf.json_format import MessageToDict
    import screen_brightness_control as sbcontrol

    pyautogui.FAILSAFE = False
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Gesture Encodings 
    class Gest(IntEnum):
        # Binary Encoded
        FIST = 0
        PINKY = 1
        RING = 2
        MID = 4
        LAST3 = 7
        INDEX = 8
        FIRST2 = 12
        LAST4 = 15
        THUMB = 16    
        PALM = 31
        
        # Extra Mappings
        V_GEST = 33
        TWO_FINGER_CLOSED = 34
        PINCH_MAJOR = 35
        PINCH_MINOR = 36

    # Multi-handedness Labels
    class HLabel(IntEnum):
        MINOR = 0
        MAJOR = 1

    # Convert Mediapipe Landmarks to recognizable Gestures
    class HandRecog:
        
        def __init__(self, hand_label):
            self.finger = 0
            self.ori_gesture = Gest.PALM
            self.prev_gesture = Gest.PALM
            self.frame_count = 0
            self.hand_result = None
            self.hand_label = hand_label
        
        def update_hand_result(self, hand_result):
            self.hand_result = hand_result

        def get_signed_dist(self, point):
            sign = -1
            if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
                sign = 1
            dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
            dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
            dist = math.sqrt(dist)
            return dist*sign
        
        def get_dist(self, point):
            dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
            dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
            dist = math.sqrt(dist)
            return dist
        
        def get_dz(self,point):
            return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
        
        # Function to find Gesture Encoding using current finger_state.
        # Finger_state: 1 if finger is open, else 0
        def set_finger_state(self):
            if self.hand_result == None:
                return

            points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
            self.finger = 0
            self.finger = self.finger | 0 #thumb
            for idx,point in enumerate(points):
                
                dist = self.get_signed_dist(point[:2])
                dist2 = self.get_signed_dist(point[1:])
                
                try:
                    ratio = round(dist/dist2,1)
                except:
                    ratio = round(dist1/0.01,1)

                self.finger = self.finger << 1
                if ratio > 0.5 :
                    self.finger = self.finger | 1
        

        # Handling Fluctations due to noise
        def get_gesture(self):
            if self.hand_result == None:
                return Gest.PALM

            current_gesture = Gest.PALM
            if self.finger in [Gest.LAST3,Gest.LAST4] and self.get_dist([8,4]) < 0.05:
                if self.hand_label == HLabel.MINOR :
                    current_gesture = Gest.PINCH_MINOR
                else:
                    current_gesture = Gest.PINCH_MAJOR

            elif Gest.FIRST2 == self.finger :
                point = [[8,12],[5,9]]
                dist1 = self.get_dist(point[0])
                dist2 = self.get_dist(point[1])
                ratio = dist1/dist2
                if ratio > 1.7:
                    current_gesture = Gest.V_GEST
                else:
                    if self.get_dz([8,12]) < 0.1:
                        current_gesture =  Gest.TWO_FINGER_CLOSED
                    else:
                        current_gesture =  Gest.MID
                
            else:
                current_gesture =  self.finger
            
            if current_gesture == self.prev_gesture:
                self.frame_count += 1
            else:
                self.frame_count = 0

            self.prev_gesture = current_gesture

            if self.frame_count > 4 :
                self.ori_gesture = current_gesture
            return self.ori_gesture

    # Executes commands according to detected gestures
    class Controller:
        tx_old = 0
        ty_old = 0
        trial = True
        flag = False
        grabflag = False
        pinchmajorflag = False
        pinchminorflag = False
        pinchstartxcoord = None
        pinchstartycoord = None
        pinchdirectionflag = None
        prevpinchlv = 0
        pinchlv = 0
        framecount = 0
        prev_hand = None
        pinch_threshold = 0.3
        
        def getpinchylv(hand_result):
            dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
            return dist

        def getpinchxlv(hand_result):
            dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
            return dist
        
        def changesystembrightness():
            currentBrightnessLv = sbcontrol.get_brightness()/100.0
            currentBrightnessLv += Controller.pinchlv/50.0
            if currentBrightnessLv > 1.0:
                currentBrightnessLv = 1.0
            elif currentBrightnessLv < 0.0:
                currentBrightnessLv = 0.0       
            sbcontrol.fade_brightness(int(100*currentBrightnessLv) , start = sbcontrol.get_brightness())
        
        def changesystemvolume():
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            currentVolumeLv = volume.GetMasterVolumeLevelScalar()
            currentVolumeLv += Controller.pinchlv/50.0
            if currentVolumeLv > 1.0:
                currentVolumeLv = 1.0
            elif currentVolumeLv < 0.0:
                currentVolumeLv = 0.0
            volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
        
        def scrollVertical():
            pyautogui.scroll(120 if Controller.pinchlv>0.0 else -120)
            
        
        def scrollHorizontal():
            pyautogui.keyDown('shift')
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(-120 if Controller.pinchlv>0.0 else 120)
            pyautogui.keyUp('ctrl')
            pyautogui.keyUp('shift')

        # Locate Hand to get Cursor Position
        # Stabilize cursor by Dampening
        def get_position(hand_result):
            point = 9
            position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
            sx,sy = pyautogui.size()
            x_old,y_old = pyautogui.position()
            x = int(position[0]*sx)
            y = int(position[1]*sy)
            if Controller.prev_hand is None:
                Controller.prev_hand = x,y
            delta_x = x - Controller.prev_hand[0]
            delta_y = y - Controller.prev_hand[1]

            distsq = delta_x**2 + delta_y**2
            ratio = 1
            Controller.prev_hand = [x,y]

            if distsq <= 25:
                ratio = 0
            elif distsq <= 900:
                ratio = 0.07 * (distsq ** (1/2))
            else:
                ratio = 2.1
            x , y = x_old + delta_x*ratio , y_old + delta_y*ratio
            return (x,y)

        def pinch_control_init(hand_result):
            Controller.pinchstartxcoord = hand_result.landmark[8].x
            Controller.pinchstartycoord = hand_result.landmark[8].y
            Controller.pinchlv = 0
            Controller.prevpinchlv = 0
            Controller.framecount = 0

        # Hold final position for 5 frames to change status
        def pinch_control(hand_result, controlHorizontal, controlVertical):
            if Controller.framecount == 5:
                Controller.framecount = 0
                Controller.pinchlv = Controller.prevpinchlv

                if Controller.pinchdirectionflag == True:
                    controlHorizontal() #x

                elif Controller.pinchdirectionflag == False:
                    controlVertical() #y

            lvx =  Controller.getpinchxlv(hand_result)
            lvy =  Controller.getpinchylv(hand_result)
                
            if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
                Controller.pinchdirectionflag = False
                if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                    Controller.framecount += 1
                else:
                    Controller.prevpinchlv = lvy
                    Controller.framecount = 0

            elif abs(lvx) > Controller.pinch_threshold:
                Controller.pinchdirectionflag = True
                if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                    Controller.framecount += 1
                else:
                    Controller.prevpinchlv = lvx
                    Controller.framecount = 0

        def handle_controls(gesture, hand_result):        
            x,y = None,None
            if gesture != Gest.PALM :
                x,y = Controller.get_position(hand_result)
            
            # flag reset
            if gesture != Gest.FIST and Controller.grabflag:
                Controller.grabflag = False
                pyautogui.mouseUp(button = "left")

            if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
                Controller.pinchmajorflag = False

            if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
                Controller.pinchminorflag = False

            # implementation
            if gesture == Gest.V_GEST:
                Controller.flag = True
                pyautogui.moveTo(x, y, duration = 0.1)

            elif gesture == Gest.FIST:
                if not Controller.grabflag : 
                    Controller.grabflag = True
                    pyautogui.mouseDown(button = "left")
                pyautogui.moveTo(x, y, duration = 0.1)

            elif gesture == Gest.MID and Controller.flag:
                pyautogui.click()
                Controller.flag = False

            elif gesture == Gest.INDEX and Controller.flag:
                pyautogui.click(button='right')
                Controller.flag = False

            elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
                pyautogui.doubleClick()
                Controller.flag = False

            elif gesture == Gest.PINCH_MINOR:
                if Controller.pinchminorflag == False:
                    Controller.pinch_control_init(hand_result)
                    Controller.pinchminorflag = True
                Controller.pinch_control(hand_result,Controller.scrollHorizontal, Controller.scrollVertical)
            
            elif gesture == Gest.PINCH_MAJOR:
                if Controller.pinchmajorflag == False:
                    Controller.pinch_control_init(hand_result)
                    Controller.pinchmajorflag = True
                Controller.pinch_control(hand_result,Controller.changesystembrightness, Controller.changesystemvolume)
            
    '''
    ----------------------------------------  Main Class  ----------------------------------------
        Entry point of Gesture Controller
    '''


    class GestureController:
        gc_mode = 0
        cap = None
        CAM_HEIGHT = None
        CAM_WIDTH = None
        hr_major = None # Right Hand by default
        hr_minor = None # Left hand by default
        dom_hand = True

        def __init__(self):
            GestureController.gc_mode = 1
            GestureController.cap = cv2.VideoCapture(0)
            GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        def classify_hands(results):
            left , right = None,None
            try:
                handedness_dict = MessageToDict(results.multi_handedness[0])
                if handedness_dict['classification'][0]['label'] == 'Right':
                    right = results.multi_hand_landmarks[0]
                else :
                    left = results.multi_hand_landmarks[0]
            except:
                pass

            try:
                handedness_dict = MessageToDict(results.multi_handedness[1])
                if handedness_dict['classification'][0]['label'] == 'Right':
                    right = results.multi_hand_landmarks[1]
                else :
                    left = results.multi_hand_landmarks[1]
            except:
                pass
            
            if GestureController.dom_hand == True:
                GestureController.hr_major = right
                GestureController.hr_minor = left
            else :
                GestureController.hr_major = left
                GestureController.hr_minor = right

        def start(self):
            
            handmajor = HandRecog(HLabel.MAJOR)
            handminor = HandRecog(HLabel.MINOR)

            with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                while GestureController.cap.isOpened() and GestureController.gc_mode:
                    success, image = GestureController.cap.read()

                    if not success:
                        print("Ignoring empty camera frame.")
                        continue
                    
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:                   
                        GestureController.classify_hands(results)
                        handmajor.update_hand_result(GestureController.hr_major)
                        handminor.update_hand_result(GestureController.hr_minor)

                        handmajor.set_finger_state()
                        handminor.set_finger_state()
                        gest_name = handminor.get_gesture()

                        if gest_name == Gest.PINCH_MINOR:
                            Controller.handle_controls(gest_name, handminor.hand_result)
                        else:
                            gest_name = handmajor.get_gesture()
                            Controller.handle_controls(gest_name, handmajor.hand_result)
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        Controller.prev_hand = None
                    cv2.imshow('Gesture Controller', image)
                    if cv2.waitKey(5) & 0xFF == 13:
                        break
            GestureController.cap.release()
            cv2.destroyAllWindows()

    # uncomment to run directly
    gc1 = GestureController()
    gc1.start()





   
    return render(request,'index.html')



def new_page_view(request):
    return render(request,'index2.html')

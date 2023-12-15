"""
File: main.py
Author: Xianhe Zhang
Email: zhang.xianh@northeastern.edu
Description: This file will be able to use different models to classify hand gestures, live video or image format.
Created: 2023-12-14
"""

import cv2
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

CURRENT_PATH = os.getcwd()
MODEL_PATH = f"{CURRENT_PATH}/model/gesture_recognizer.task"
GAME_MODEL_PATH = f"{CURRENT_PATH}/game_model/gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))



# The function helps to draw all the feature points, and 
def process_and_display_frame(image, hand_landmarks) -> None:
    # Draw hand landmarks
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )




def main():
    argc = len(sys.argv) 
    argv = sys.argv     
    
    if argc == 3:
        base_options = python.BaseOptions(model_asset_path=GAME_MODEL_PATH)
    elif argc > 3:
        exit(-1) 
    else:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode = VisionRunningMode.IMAGE)
    hand_drawing_utils = mp.solutions.drawing_utils
    recognizer = vision.GestureRecognizer.create_from_options(options)  

    # Init text postion, fond, size, color and bold.
    org = (50, 50)  
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 1  
    color = (255, 0, 0) 
    thickness = 2 


    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    if not cap.isOpened():
        print("unable to open video device, please try again")
        exit(-1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Expected Size: {width} X {height}')

    cv2.namedWindow('Video')
    i_ret, i_img = cap.read()
    cv2.imshow("Video", i_img)
    hc = set()
    
    while True:
        if argc == 2:
            img = cv2.imread(argv[1])
            if img is None:
                print("input image is empty")
                break
        else: 
            ret, img = cap.read()
            if not ret: 
                print("frame is empty")
                break


        # Convert cam frame into certian format used by recognizer. 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        # Use the model to classify/recoginize.
        recognizer_result = recognizer.recognize(mp_image)

        # If specific type of hand gesture is identified, then we need to draw the picture and show.
        if recognizer_result:
            if recognizer_result.gestures and recognizer_result.hand_landmarks:    
                top_gesture = recognizer_result.gestures[0][0].category_name 
                hand_landmarks = recognizer_result.hand_landmarks[0]

                if top_gesture in ("Closed_Fist","Open_Palm","Pointing_Up","Thumb_Down","Thumb_Up","Victory","ILoveYou","scissors","paper","rock","rocks"):
                    print(f"gesture name: {top_gesture}")
                    cv2.putText(img, top_gesture, org, font, font_scale, color, thickness)
                    process_and_display_frame(img, hand_landmarks)
                    cv2.imshow("Video", img)

                else:
                    hc.add(top_gesture)
                    print(hc)        

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Main Complete"

if __name__ == '__main__':
    main()



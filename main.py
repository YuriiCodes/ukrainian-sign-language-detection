import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                dots = list(enumerate(hand_landmarks.landmark))  # list of coordinates for each point
                # thumb
                x_big = dots[4][1].x * w
                y_big = dots[4][1].y * h
                # index finger
                x_forefinger = dots[8][1].x * w
                y_forefinger = dots[8][1].y * h
                # middle finger
                x_middle = dots[12][1].x * w
                y_middle = dots[12][1].y * h
                # ring finger
                x_ring = dots[16][1].x * w
                y_ring = dots[16][1].y * h
                # little finger
                x_little = dots[20][1].x * w
                y_little = dots[20][1].y * h
                # wrist
                x_wrist = dots[0][1].x * w
                y_wrist = dots[0][1].y * h

                # base of the thumb
                x_big_base = dots[1][1].x * w
                y_big_base = dots[1][1].y * w
                # base of the index finger
                x_forefinger_base = dots[5][1].x * w
                y_forefinger_base = dots[5][1].y * h
                # base of the middle finger
                x_middle_base = dots[9][1].x * w
                y_middle_base = dots[9][1].y * h
                # base of the ring finger
                x_ring_base = dots[13][1].x * w
                y_ring_base = dots[13][1].y * h
                # base of the little finger
                x_little_base = dots[17][1].x * w
                y_little_base = dots[17][1].y * h

                # distance between point 0 and point 17
                dist_0_17 = int(math.sqrt(pow(x_little_base - x_wrist, 2) + pow(y_little_base - y_wrist, 2)))
                # distance between thumb and index finger
                delta_thumb_forefinger = int(math.sqrt(pow(x_forefinger - x_big, 2) + pow(y_forefinger - y_big, 2)))
                # distance between thumb and middle finger
                delta_thumb_middle = int(math.sqrt(pow(x_middle - x_big, 2) + pow(y_middle - y_big, 2)))
                # distance between thumb and ring finger
                delta_thumb_ring = int(math.sqrt(pow(x_ring - x_big, 2) + pow(y_ring - y_big, 2)))
                # distance between thumb and little finger
                delta_thumb_little = int(math.sqrt(pow(x_little - x_big, 2) + pow(y_little - y_big, 2)))
                # distance between index finger and its base
                delta_forefinger_base = int(math.sqrt(pow(x_forefinger_base - x_forefinger, 2) + pow(y_forefinger_base - y_forefinger, 2)))
                # distance between thumb and its base
                delta_thumb_base = int(math.sqrt(pow(x_big - x_big_base, 2) + pow(y_big - y_big_base, 2)))
                # distance between middle finger and its base
                delta_middle_base = int(math.sqrt(pow(x_middle - x_middle_base, 2) + pow(y_middle - y_middle_base, 2)))
                # distance between the little fingertip and the wrist
                d_little_wrist = int(math.sqrt(pow(x_wrist - x_little, 2) + pow(y_wrist - y_little, 2)))
                # distance between the little fingertip and its base
                d_little_base = int(math.sqrt(pow(x_little - x_little_base, 2) + pow(y_little - y_little_base, 2)))
                # distance between the tips of the index and middle fingers
                d_foref_middl = int(math.sqrt(pow(x_forefinger - x_middle, 2) + pow(y_forefinger - y_middle, 2)))

                # percentages of the distance from the thumb tip to its base
                p_5_t_b = delta_thumb_base / 100 * 5
                p_6_t_b = delta_thumb_base / 100 * 6
                p_10_t_b = delta_thumb_base / 100 * 10

                # use the distance between points 0 and 17 as a baseline for comparison
                cv2.putText(image, f'dist_0_17 - {dist_0_17}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)

                # letter "A"
                if d_little_base < 45:
                    cv2.putText(image, 'A', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)

                # letter "Б"
                if d_foref_middl < 55 and delta_thumb_ring < 50 and delta_thumb_little < 50 and delta_thumb_forefinger > 120:
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(image, 'Б', (100, 300), font, 3, (255, 255, 255), thickness=2)

                # letter "О"
                if delta_thumb_forefinger < 32 and delta_thumb_ring > 45 and delta_thumb_middle > 45 and delta_thumb_little > 50:
                    cv2.putText(image, 'O', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)
                cv2.putText(image, f'dist_th_frf-{delta_thumb_forefinger}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)

                # letter "P"
                if delta_thumb_middle < 32 and delta_thumb_forefinger > 45 and delta_thumb_ring > 45:
                    cv2.putText(image, 'P', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)
                # letter "H"
                if delta_thumb_ring < 32 and delta_thumb_middle > 45 and delta_thumb_forefinger > 45:
                    cv2.putText(image, 'H', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)

                # letter "E"
                if delta_thumb_ring < 34 and delta_thumb_middle < 34 and delta_thumb_forefinger < 34 and delta_thumb_little < 34:
                    cv2.putText(image, 'E', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)

                # letter "Г"
                dot_3_x = dots[3][1].x * w
                dot_3_y = dots[3][1].y * h
                dot_4_x = dots[4][1].x * w
                dot_4_y = dots[4][1].y * h
                dot_5_x = dots[5][1].x * w
                dot_5_y = dots[5][1].y * h
                dot_6_x = dots[6][1].x * w
                dot_6_y = dots[6][1].y * h
                dot_12_x = dots[12][1].x * h
                dot_12_y = dots[12][1].y * h
                dot_16_x = dots[16][1].x * h
                dot_16_y = dots[16][1].y * h
                dot_20_x = dots[20][1].x * h
                dot_20_y = dots[20][1].y * h
                dot_10_x = dots[10][1].x * h
                dot_10_y = dots[10][1].y * h
                dot_14_x = dots[14][1].x * h
                dot_14_y = dots[14][1].y * h
                dot_18_x = dots[18][1].x * h
                dot_18_y = dots[18][1].y * h
                vect_43 = [dot_4_x - dot_3_x, dot_4_y - dot_3_y]
                vect_65 = [dot_6_x - dot_5_x, dot_6_y - dot_5_y]
                dis_vect_43 = math.sqrt(pow(vect_43[0], 2) + pow(vect_43[1], 2))
                dis_vect_65 = math.sqrt(pow(vect_65[0], 2) + pow(vect_65[1], 2))
                scal_vect_43_65 = (vect_43[0] * vect_65[0] + vect_43[1] * vect_65[1]) / (dis_vect_43 * dis_vect_65)
                x = np.arccos([scal_vect_43_65])
                y = x * 180 / math.pi
                cv2.putText(image, f'angle-{y}', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)
                if 55 < y < 105 and dot_12_y < dot_10_y and dot_16_y < dot_14_y and dot_20_y < dot_18_y and y_forefinger > dot_10_y:
                    cv2.putText(image, 'Г', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)


                # the tips of the other fingers are located between the higher phalanges
                # the angle is approximately 50-110 degrees
                # the tip of the index finger is lower than all the other points of the other fingers

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

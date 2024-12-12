from pathlib import Path
import cv2
import os

# Videos:
# computing_floor_full = full video in computing_floor_full/video/video.mov
# computing_floor_regular = full video in computing_floor_regular/video/video.mov
# computing_floor_vertical_angle = = full video in computing_floor_vertical_angle/video/video.mov
# computing_floor_wide_angle = = full video in computing_floor_wide_angle/video/video.mov
# computing_floor_faculty_offices = = first 70 seconds of video in computing_floor_full/video/video.mov
# computing_floor_75_wide_angle = = first 60 seconds of video in computing_floor_75_wide_angle_/video.mov
# the_green_geotagged = shot in 

# Ideal settings:
#   - grabbing 3-4 frames per second (@ 60fps, keeping line 30 "if frame_no % 15 == 0")s
#                                    (@ 30fps, keeping line 30 "if frame_no % 30 == 0")s   

video_path = Path('/Users/davidlelis/Code/CIS6900_Machine-Learning/project/the_green/video/video.MOV')
img_path = Path('/Users/davidlelis/Code/CIS6900_Machine-Learning/project/the_green/images')
img_path.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

frame_no = 0
frame_rate = 60

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    if frame_no % 15 == 0: # pick up every three frames per second (60 fps)
        frame_path = img_path / f'frame{frame_no:05d}.jpg'
        cv2.imwrite(str(frame_path), frame)

    frame_no += 1
    if frame_no > frame_rate * 304:
        break

cap.release()
print('done')
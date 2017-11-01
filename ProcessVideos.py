import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

import CameraManager
import LaneLine
import FrameProcessor

def process_image(img):
    global num_frames_global
    
    num_frames_global += 1
    processor.processFrame(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    #return cv2.cvtColor(processor.visLaneDetectImage, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(processor.visBinaryImage, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(processor.visImageAnnotated, cv2.COLOR_BGR2RGB)

def process_video(file_name, sub_clip_from = 0, sub_clip_to = 0, visualization = False):
    global num_frames_global
    num_frames_global = 0

    camera = CameraManager.CameraManager('center')
    camera.initPerspectiveTransformation(srcPoints, dstPoints, dtsPlaneSizePx, dtsPlaneSizeM)

    leftLane = LaneLine.LaneLine()
    rightLane = LaneLine.LaneLine()

    global processor
    processor = FrameProcessor.FrameProcessor(camera, leftLane, rightLane, visualization = visualization)

    v_clip = VideoFileClip(input_dir_path + file_name)
    if sub_clip_to > 0:
        v_clip = v_clip.subclip(sub_clip_from, sub_clip_to)

    white_clip = v_clip.fl_image(process_image)
    white_clip.write_videofile(output_dir_path + file_name, audio=False)
    print("Video is processed. Frames: {0}.".format(num_frames_global))
    return

input_dir_path = "./test_videos/"
output_dir_path = "./test_videos_output/"

try:
    os.makedirs(output_dir_path)
except:
    pass

srcPoints = [[246, 700], [570, 468], [715, 468], [1079, 700]]
dstPoints = [[400, 719], [400, 0], [1200, 0], [1200, 719]]
dtsPlaneSizePx = (720, 1600)
dtsPlaneSizeM = (30.0, 7.4)

process_video("challenge_video.mp4")

process_video("project_video.mp4")

process_video("harder_challenge_video.mp4")

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import CameraManager
import LaneLine
import FrameProcessor

def processFrame(src_name, visualization = True, show_diagram = False):
    img = cv2.imread(input_dir_path + src_name)

    camera = CameraManager.CameraManager('center')
    camera.initPerspectiveTransformation(srcPoints, dstPoints, dtsPlaneSizePx, dtsPlaneSizeM)

    leftLane = LaneLine.LaneLine()
    rightLane = LaneLine.LaneLine()
    processor = FrameProcessor.FrameProcessor(camera, leftLane, rightLane, visualization = visualization)
    processor.processFrame(img)

    if visualization:
        cv2.imwrite(undist_dir_path+src_name, processor.visUndistortImage)
        cv2.imwrite(binary_dir_path+src_name, processor.visBinaryImage)
        cv2.imwrite(topview_dir_path+src_name, processor.visTopViewBinaryImage)
        cv2.imwrite(bindetect_dir_path+src_name, processor.visLaneDetectImage)

    cv2.imwrite(annotated_dir_path+src_name, processor.visImageAnnotated)

    if show_diagram:
        # Visualize img binary
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(cv2.cvtColor(processor.visLaneDetectImage, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize = 30)
        ax2.imshow(cv2.cvtColor(processor.visImageAnnotated, cv2.COLOR_BGR2RGB))
        ax2.set_title('Binary Image', fontsize = 30)
        plt.show()

input_dir_path = "./test_images/"
output_dir_path = "./test_images_output/"
undist_dir_path = "./test_images_output/undist_images/"
binary_dir_path = "./test_images_output/binary_images/"
topview_dir_path = "./test_images_output/topview_images/"
bindetect_dir_path = "./test_images_output/bindetect_images/"
annotated_dir_path = "./test_images_output/annotated_images/"

srcPoints = [[246, 700], [570, 468], [715, 468], [1079, 700]]
dstPoints = [[400, 719], [400, 0], [1200, 0], [1200, 719]]
dtsPlaneSizePx = (720, 1600)
dtsPlaneSizeM = (30.0, 7.4)

images_dir = None
try: images_dir = os.listdir(input_dir_path)
except: print("Cannot read list of images")

try: os.makedirs(output_dir_path)
except: pass
try: os.makedirs(undist_dir_path)
except: pass
try: os.makedirs(binary_dir_path)
except: pass
try: os.makedirs(topview_dir_path)
except: pass
try: os.makedirs(bindetect_dir_path)
except: pass
try: os.makedirs(annotated_dir_path)
except: pass

if images_dir is not None:
    for image_name in images_dir:
        try:
            image_path = input_dir_path + image_name
            if os.path.exists(image_path) and os.path.isfile(image_path):
                print("Processing image {0} ...".format(image_name))
                processFrame(image_name)
                print("    Done.")
        except:
            print("Image {0} cannot be processed.".format(image_name))

        print()


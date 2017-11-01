## **Advanced Lane Finding**

Detect lane lines on the road in real time, estimate it curvature and vehicle position in respect to lane center.

![Advanced Lane Finding](./images/main.jpg)

---

Practical project to recognize road marking from images and video streams, highlight (annotate) left and right edges of the road lane, calculate it curvature and estimate car position. Project was done on Python. More details are [here](./writeup.md).

**Project content**
*	[CameraManager.py](./CameraManager.py) – Python class, used to manage camera parameters and perform image transformations.
*	[FrameProcessor.py](./FrameProcessor.py) – Python class, implements main lane line recognition pipeline.
*	[LaneLine.py](./LaneLine.py) – Python class, contains information specific for separate lane line.
*	[CameraCalibration.py](./CameraCalibration.py) – responsible for camera calibration. Run it to calibrate camera.
*	[ProcessImages.py](./LaneLine.py) – run image pipeline.
*	[ProcessVideos.py](./ProcessVideos.py) – run video pipeline.
*	[writeup.md](./writeup.md) – project description in details
*	[README.md](./README.md) – this file.
* [camera](./camera) - folder, contains camera parameters (results of calibration) automatically loaded by [CameraManager](./CameraManager.py) class.
* [camera_cal](./camera_cal) - folder with images used for camera calibration.
* [camera_undist](./camera_undist) - folder contains examples of undistorted images, result of camera calibration test.
* [images](./images) - folder with different images used in project writeup.
*	[test_images](./test_images) – folder with test source images
*	[output_images](./output_images) - folder with annotated images
*	[test_videos](./test_videos) - folder with test video files
*	[test_videos_output](./test_videos_output) - folder with annotated video files

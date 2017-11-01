import numpy as np
import cv2
import LaneLine

class FrameProcessor():
    """
    Frame processor - implement pipeline to detect lane lines in frame
    """

    def __init__(
        self,
        camera, # Camera instance
        leftLane, # LaneLine instance for left lane detection
        rightLane, # LaneLine instance for right lane detection
        visualization = False, # If this parameter is enabled - produce vizualization of frame processing step by step.
        detectorInitSize = (0.3, 0.3), # Window size to detect initial bottom position of lane line - (height, width)
        detectorWinSize = (20, 50, 180), # (Height, Bottom width, Top width) - convolution windows size
        detectorWinMarging = (100, 50, 180), # (Initial bottom, Bottom, Top) - ranges where convolution operation will be applied
        detectorEmptyThreshold = 0.1, # Threshold to recognize window as empty, range [0..1]
        detectorEmptySeq = 10, # Maximal number of consecutive empty windows to stop lane detection
        detectorMinWinPerLine = 8,  # Minimal number of windows must be detected for successfull recognition
        detectorMinApproxBoxes = 5, # Minimal number of windows detected to start lane approximation
        detectorMinApproxBoxesSq = 7, # Minimal number of windows detected to start curve approximation
        detectorMutualLaneShapeCorrection = False, # Use mutual lane shape correction or not
        detectorUsePreviousLine = True, # Use previous line shape as start point
        detectorWeightFactor = 4.0, # Detector weight factor - weight of bottom pixels in comparison to top pixels have always weight 1.0
        detectorMaxAllowedCurvDiff = 6.0, # Maximal allowed cuvrature difference between lines. This is logarithmic empirical value
        annotationColorBGR = (0, 255, 0), # Annotation BGR color for space between lines
        annotationColorLineLeftBGR = (0, 0, 255), # Annotation BGR color for left line
        annotationColorLineRightBGR = (255, 0, 0), # Annotation BGR color for right line
        annotationColorTextBGR = (255, 255, 255), # Annotation BGR color for text
        annotationWeight = 0.3, # Transparence factor for annotation lanes and space between. In comparison to main image.
        annotationDelayFrames = 7, # Delay of annotation information refresh (in video frames)
        annotationBufferSize = 15 # Size of annotation buffer for averaging (in video frames)
        ):
        
        self.camera = camera
        self.laneLines = (leftLane, rightLane)

        self.detectorInitSize = detectorInitSize
        self.detectorWinSize = detectorWinSize
        self.detectorWinMarging = detectorWinMarging
        self.detectorEmptyThreshold = detectorEmptyThreshold
        self.detectorEmptySeq = detectorEmptySeq
        self.detectorMinWinPerLine = detectorMinWinPerLine
        self.detectorMinApproxBoxes = detectorMinApproxBoxes
        self.detectorMinApproxBoxesSq = detectorMinApproxBoxesSq
        self.detectorMutualLaneShapeCorrection = detectorMutualLaneShapeCorrection
        self.detectorUsePreviousLine = detectorUsePreviousLine
        self.detectorWeightFactor = detectorWeightFactor
        self.detectorMaxAllowedCurvDiff = detectorMaxAllowedCurvDiff
        self.annotationColorBGR = annotationColorBGR
        self.annotationColorLineLeftBGR = annotationColorLineLeftBGR
        self.annotationColorLineRightBGR = annotationColorLineRightBGR
        self.annotationColorTextBGR = annotationColorTextBGR
        self.annotationWeight = annotationWeight
        self.annotationDelayFrames = annotationDelayFrames
        self.annotationBufferSize = annotationBufferSize
        
        self.visualization = visualization
        self.visOrigImage = None
        self.visUndistortImage = None
        self.visBinaryImage = None
        self.visTopViewBinaryImage = None
        self.visLaneDetectImage = None
        
        self.isImageAnnotated = False
        self.visImageAnnotated = None
        self.annotationRadiusBuf = []
        self.annotationRadius = 0
        self.annotationCenterShiftBuf = []
        self.annotationCenterShift = 0
        self.annotationDelayCnt = -1

    def _detectLines(
        self,
        binary # Binary top view matrix 
        ):
        """
        Detect lane lines - main algorithm of lane line detector on binary top-view image
        """

        # Adjust windows size to not exceed image shape
        def make_win_adj(win):
            return (
                int(min(bin_shape[0], max(0, win[0]))),
                int(min(bin_shape[1], max(0, win[1]))),
                int(min(bin_shape[0], max(0, win[2]))),
                int(min(bin_shape[1], max(0, win[3]))))

        # Process windows - do vertical sum, convolve and calculate fulfillment of window
        def process_win(win):
            win_adj = make_win_adj(win)

            if (win_adj[3] - win_adj[1]) > conv_win_size:
                vert_sum = np.sum(binary[win_adj[0]:win_adj[2], win_adj[1]:win_adj[3]], axis=0)
                win_center = np.argmax(np.convolve(vert_sum, conv_win, mode = 'valid')) + win_adj[1] + conv_win_half
            else:
                win_center = (win_adj[1] + win_adj[3]) // 2

            win_adj = make_win_adj((win_adj[0], win_center - conv_win_half, win_adj[2], win_center + conv_win_half))
            win_sum = np.sum(binary[win_adj[0]:win_adj[2], win_adj[1]:win_adj[3]])
            win_sq = (win_adj[2] - win_adj[0]) * (win_adj[3] - win_adj[1])

            if win_sq > 0:
                win_pct = float(win_sum) / float(win_sq)
            else:
                win_pct = -1.0

            return win_adj, win_center, win_sum, win_pct

        # Convert set of windows to array of points these windows contain
        def combine_line_points(win_matrix, color_plane_bgr):
            line_points = []

            if len(win_matrix) >= self.detectorMinWinPerLine:
                for y1, x1, y2, x2 in win_matrix:
                    idx_grid = np.mgrid[y1:y2, x1:x2]
                    idx_filter = np.array(binary[y1:y2, x1:x2], dtype=bool)
                    points = idx_grid[:, idx_filter]
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)
                    points = np.append(points, [points_w], axis=0).T

                    # Vizualize windows
                    if self.visualization:
                        self.visLaneDetectImage[y1:y2, x1:x2] = self.visLaneDetectImage[y1:y2, x1:x2] * 0.75
                        self.visLaneDetectImage[y1:y2, x1:x2, color_plane_bgr] = 255
                    
                    if len(line_points) > 0:
                        line_points = np.append(line_points, points, axis=0)
                    else:
                        line_points = points

            return line_points

        # Vizualuze lane lines if any detected
        def visualize_lane_line(lane_line, color_bgr = (0, 255, 255), thickness = 5):
            if self.visualization:
                if lane_line.isLineDetected:
                    vect_y = np.array(np.mgrid[0:bin_shape[0]], dtype = np.float64)
                    vect_x = lane_line.lineShape[0] * (vect_y**2) + lane_line.lineShape[1] * vect_y + lane_line.lineShape[2]

                    vect_filter = (vect_x >= 0) & (vect_x < bin_shape[1])
                    vect_y = vect_y[vect_filter]
                    vect_x = vect_x[vect_filter]

                    points = np.array([vect_x, vect_y], dtype = np.int32).T

                    cv2.polylines(self.visLaneDetectImage, [points], 0, color_bgr, thickness = thickness)

        # Vizualuze history points to see total data set for videos
        def visualize_lane_points(lane_line, color_plane_bgr):
            if self.visualization:
                for points in lane_line.histPoints:
                    if len(points) > 0:
                        for y1, x1, w in points:
                            self.visLaneDetectImage[y1, x1, color_plane_bgr] = 255

        if self.visualization:
            self.visLaneDetectImage = self.camera.binaryToImg(binary)

        bin_shape = binary.shape

        window_matrix_l = [] # (x1, y1, x2, y2) - coordinates of windowses related to left lane line
        window_matrix_r = [] # (x1, y1, x2, y2) - coordinates of windowses related to right lane line
        conv_win_size = self.detectorWinSize[1]
        conv_win_half = conv_win_size // 2 # Half of convolution window size
        conv_win = np.ones(conv_win_size) # Convolution windows - contains all values 1.0
        detectorWinMarging = self.detectorWinMarging[1]

        # First step - roughly detect bottom start position for both lines
        # For this purpose we select 2 windows at the bottom of the screen centered by left and right image halfs
    
        win_pos_x_w = int(bin_shape[1] * self.detectorInitSize[1])
        win_pos_x_l = int(((bin_shape[1] / 2.0) - win_pos_x_w) / 2.0)
        if win_pos_x_l < 0:
            win_pos_x_l = 0

        win_pos_x_r = int((bin_shape[1] / 2.0) + win_pos_x_l)

        win_pos_y = int(bin_shape[0] * (1.0 - self.detectorInitSize[0]))
        img_pos_y = bin_shape[0]

        if self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
            # Define start point with lane line detected from previous video frames
            l_center = self.laneLines[0].lineShape[0] * (img_pos_y ** 2) + self.laneLines[0].lineShape[1] * img_pos_y + self.laneLines[0].lineShape[2]
        else:
            # Define start point with convolution of bottom left and right part of image.
            win_adj, l_center, win_sum, win_pct = process_win((win_pos_y, win_pos_x_l, bin_shape[0], win_pos_x_l + win_pos_x_w))

        if self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
            # Define start point with lane line detected from previous video frames
            r_center = self.laneLines[1].lineShape[0] * (img_pos_y ** 2) + self.laneLines[1].lineShape[1] * img_pos_y + self.laneLines[1].lineShape[2]
        else:
            # Define start point with convolution of bottom left and right part of image.
            win_adj, r_center, win_sum, win_pct = process_win((win_pos_y, win_pos_x_r, bin_shape[0], win_pos_x_r + win_pos_x_w))

        det_marging_l = self.detectorWinMarging[0]
        l_detect = True
        l_empty = 0
        l_center_points = []
        
        det_marging_r = self.detectorWinMarging[0]
        r_detect = True
        r_empty = 0
        r_center_points = []

        while (l_detect or r_detect) and (img_pos_y > 0):
            # Left line detection
            if l_detect:
                # Extrapolation to detect next window center
                l_shape = []
                if len(l_center_points) > self.detectorMinApproxBoxes:
                    points = np.array(l_center_points, dtype = np.int32).T
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)

                    if len(window_matrix_l) >= self.detectorMinApproxBoxesSq:
                        l_shape = np.polyfit(points[0], points[1], 2, w = points_w)
                    else:
                        l_shape = np.polyfit(points[0], points[1], 1, w = points_w)
                elif self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
                    l_shape = self.laneLines[0].lineShape

                if len(l_shape) >= 3:
                    l_center = l_shape[0] * (img_pos_y ** 2) + l_shape[1] * img_pos_y + l_shape[2]
                    det_marging_l = detectorWinMarging
                elif len(l_shape) >= 2:
                    l_center = l_shape[0] * img_pos_y + l_shape[1]
                    det_marging_l = detectorWinMarging

                if ((l_center - conv_win_half) < 0) | ((l_center + conv_win_half) >= bin_shape[1]):
                    # If left or right side of image reached, stop detection process to avoid lane line deformation
                    l_detect = False
                else:
                    # Use convolution to detect next segment of lane line
                    win_adj_l, l_center_new, win_sum, win_pct = process_win(
                        (img_pos_y - self.detectorWinSize[0],
                         l_center - conv_win_half - det_marging_l,
                         img_pos_y,
                         l_center + conv_win_half + det_marging_l))

                    if win_pct >= self.detectorEmptyThreshold:
                        l_empty = 0
                        l_center = l_center_new
                        window_matrix_l += [win_adj_l]
                        l_center_points += [[img_pos_y, l_center_new]]
                    else:
                        l_empty += 1
                        l_center_points += [[img_pos_y, l_center]]
                        #if l_empty >= self.detectorEmptySeq:
                        #    l_detect = False

            # Right line detection
            if r_detect:
                # Extrapolation to detect next window center
                r_shape = []
                if len(r_center_points) > self.detectorMinApproxBoxes:
                    points = np.array(r_center_points, dtype = np.int32).T
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)

                    if len(window_matrix_r) >= self.detectorMinApproxBoxesSq:
                        r_shape = np.polyfit(points[0], points[1], 2, w = points_w)
                    else:
                        r_shape = np.polyfit(points[0], points[1], 1, w = points_w)
                elif self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
                    r_shape = self.laneLines[1].lineShape

                if len(r_shape) >= 3:
                    r_center = r_shape[0] * (img_pos_y ** 2) + r_shape[1] * img_pos_y + r_shape[2]
                    det_marging_r = detectorWinMarging
                elif len(r_shape) >= 2:
                    r_center = r_shape[0] * img_pos_y + r_shape[1]
                    det_marging_r = detectorWinMarging

                if ((r_center - conv_win_half) < 0) | ((r_center + conv_win_half) >= bin_shape[1]):
                    # If left or right side of image reached, stop detection process to avoid lane line deformation
                    r_detect = False
                else:
                    # Use convolution to detect next segment of lane line
                    win_adj_r, r_center_new, win_sum, win_pct = process_win(
                        (img_pos_y - self.detectorWinSize[0],
                         r_center - conv_win_half - det_marging_r,
                         img_pos_y,
                         r_center + conv_win_half + det_marging_r))

                    if win_pct >= self.detectorEmptyThreshold:
                        r_empty = 0
                        r_center = r_center_new
                        window_matrix_r += [win_adj_r]
                        r_center_points += [[img_pos_y, r_center_new]]
                    else:
                        r_empty += 1
                        r_center_points += [[img_pos_y, r_center]]
                        #if r_empty >= self.detectorEmptySeq:
                        #    r_detect = False

            img_pos_y -= self.detectorWinSize[0]

            if img_pos_y > 0:
                # Calculate size of next convolution window. This size is increasing from bottom to top of the image.
                conv_win_size_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinSize[2] - self.detectorWinSize[1])))
                detectorWinMarging_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinMarging[2] - self.detectorWinMarging[1])))

                if conv_win_size_new != conv_win_size:
                    conv_win_size = conv_win_size_new
                    conv_win_half = conv_win_size // 2
                    conv_win = np.ones(conv_win_size)

                detectorWinMarging_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinMarging[2] - self.detectorWinMarging[1])))
                if detectorWinMarging_new != detectorWinMarging:
                    detectorWinMarging = detectorWinMarging_new

            det_marging_l = detectorWinMarging
            det_marging_r = detectorWinMarging

        # Retrive lane points from set of windows
        line_points_l = combine_line_points(window_matrix_l, 0)
        line_points_r = combine_line_points(window_matrix_r, 2)

        # Validate lane lines separately
        valid_pos_y = bin_shape[0] - 1
        checkCurvaturePoints = np.mgrid[0:bin_shape[0]]

        if self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
            l_shape = self.laneLines[0].lineShape
            l_center = l_shape[0] * (valid_pos_y ** 2) + l_shape[1] * valid_pos_y + l_shape[2]
            check_res_l, line_points_l, line_shape_l = self.laneLines[0].checkFilterLinePoints(line_points_l, (bin_shape[0], max(win_pos_x_l, l_center - self.detectorWinMarging[0]), min(win_pos_x_l + win_pos_x_w, l_center + self.detectorWinMarging[0])), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)
        else:
            check_res_l, line_points_l, line_shape_l = self.laneLines[0].checkFilterLinePoints(line_points_l, (valid_pos_y, win_pos_x_l, win_pos_x_l + win_pos_x_w), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)

        if self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
            r_shape = self.laneLines[1].lineShape
            r_center = r_shape[0] * (valid_pos_y ** 2) + r_shape[1] * valid_pos_y + r_shape[2]
            check_res_r, line_points_r, line_shape_r = self.laneLines[1].checkFilterLinePoints(line_points_r, (bin_shape[0], max(win_pos_x_r, r_center - self.detectorWinMarging[0]), min(win_pos_x_r + win_pos_x_w, r_center + self.detectorWinMarging[0])), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)
        else:
            check_res_r, line_points_r, line_shape_r = self.laneLines[1].checkFilterLinePoints(line_points_r, (valid_pos_y, win_pos_x_r, win_pos_x_r + win_pos_x_w), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)

        # Validate both lane lines
        addPoints = False
        if check_res_l and check_res_r:
            # Validate if lane lines intersect within double image height.
            a1 = line_shape_l[0] - line_shape_r[0]
            b1 = line_shape_l[1] - line_shape_r[1]
            c1 = line_shape_l[2] - line_shape_r[2]
            det = b1 ** 2 - 4 * a1 * c1
            if det >= 0:
                y1 = (-b1 - np.sqrt(det)) / (2 * a1)
                y2 = (-b1 + np.sqrt(det)) / (2 * a1)
            else:
                y1 = 0
                y2 = 0

            if (det < 0) or ((abs(y1) > bin_shape[0]) and (abs(y2) > bin_shape[0])):
                # Validate if lane line curvature is comparable
                if (line_shape_l[0] != 0) and (line_shape_r[0] != 0):
                    rad1 = np.log10(np.min(((1 + (2 * line_shape_l[0] * checkCurvaturePoints + line_shape_l[1]) ** 2) ** 1.5) / np.absolute(2 * line_shape_l[0])))
                    rad2 = np.log10(np.min(((1 + (2 * line_shape_r[0] * checkCurvaturePoints + line_shape_r[1]) ** 2) ** 1.5) / np.absolute(2 * line_shape_r[0])))

                    if (rad1 != 0) and (rad2 != 0):
                        change_power = 100.0**(max(rad1, rad2) / min(rad1, rad2)) / 100.0
                        if change_power <= self.detectorMaxAllowedCurvDiff:
                            addPoints = True
                else:
                    addPoints = True

        # Add detected points to lane line history. If validation was not successful, empty values are added
        if addPoints:
            self.laneLines[0].addLinePoints(line_points_l)
            self.laneLines[1].addLinePoints(line_points_r)
        else:
            self.laneLines[0].addLinePoints([])
            self.laneLines[1].addLinePoints([])

        # Update shape of lane lines
        world_correction = self.camera.getWorldCorrection()
        self.laneLines[0].updateLineShape(world_correction)
        self.laneLines[1].updateLineShape(world_correction)

        # if mutual lane shape correction enabled, do correction to calculate average lane lines shape for both lines
        if self.detectorMutualLaneShapeCorrection:
            if self.visualization:
                visualize_lane_line(self.laneLines[0], thickness = 2)
                visualize_lane_line(self.laneLines[1], thickness = 2)

            # Lane shape must be approxumately the same, so we can mutually correct it both
            fix_point_y = bin_shape[0] * (1.0 - self.detectorInitSize[0])
            fix_point_world_y = fix_point_y * world_correction[0]
            LaneLine.LaneLine.mutualLaneShapeCorrection(self.laneLines[0], self.laneLines[1], fix_point_y, fix_point_world_y)

        # Perform vizualization
        if self.visualization:
            visualize_lane_line(self.laneLines[0], thickness = 7)
            visualize_lane_line(self.laneLines[1], thickness = 7)

        if self.visualization:
            visualize_lane_points(self.laneLines[0], 1)
            visualize_lane_points(self.laneLines[1], 1)

    def _calculateParameters(
        self,
        img, # Image for final vizualization
        top_shape # Shape of top-view image
        ):
        """
        Calculate parameters based on detected lane lines.
        """

        if (not self.laneLines[0].isLineDetected) or (not self.laneLines[1].isLineDetected):
            return False, img

        res_img = img.copy()
        img_shape = res_img.shape

        # Create set of points from lane line shape and perform inverse perspective transformation
        vect_y = np.array(np.mgrid[0:top_shape[0]], dtype = np.float64)
        vect_x1 = self.laneLines[0].lineShape[0] * (vect_y**2) + self.laneLines[0].lineShape[1] * vect_y + self.laneLines[0].lineShape[2]
        vect_x2 = self.laneLines[1].lineShape[0] * (vect_y**2) + self.laneLines[1].lineShape[1] * vect_y + self.laneLines[1].lineShape[2]

        points1 = np.array([vect_x1, vect_y], dtype = np.int32).T
        points1 = np.array(self.camera.perspectiveTransformPoints(points1, True), dtype = np.int32)

        points2 = np.array([vect_x2, vect_y], dtype = np.int32).T
        points2 = np.array(self.camera.perspectiveTransformPoints(points2, True), dtype = np.int32)

        world_correction = self.camera.getWorldCorrection()
        checkCurvaturePoints = np.mgrid[0:top_shape[0]] * world_correction[0]

        self.annotationDelayCnt = (self.annotationDelayCnt + 1) % self.annotationDelayFrames

        # Calculate curvature of lane lines
        if self.laneLines[0].isLineDetected and (self.laneLines[0].lineShapeWorld[0] != 0):
            cur_rad1 = np.min(((1 + (2 * self.laneLines[0].lineShapeWorld[0] * checkCurvaturePoints + self.laneLines[0].lineShapeWorld[1]) ** 2) ** 1.5) / np.absolute(2 * self.laneLines[0].lineShapeWorld[0]))
            # Straight lines can have different values, so makes sense limit maximum value.
            cur_rad1 = min(5000, cur_rad1)
        else:
            cur_rad1 = 0

        if self.laneLines[1].isLineDetected and (self.laneLines[1].lineShapeWorld[0] != 0):
            cur_rad2 = np.min(((1 + (2 * self.laneLines[1].lineShapeWorld[0] * checkCurvaturePoints + self.laneLines[1].lineShapeWorld[1]) ** 2) ** 1.5) / np.absolute(2 * self.laneLines[1].lineShapeWorld[0]))
            # Straight lines can have different values, so makes sense limit maximum value.
            cur_rad1 = min(5000, cur_rad1)
        else:
            cur_rad2 = 0

        if (cur_rad1 <= 0) & (cur_rad2 <= 0):
            cur_rad = 0
        elif cur_rad1 <= 0:
            cur_rad = cur_rad2
        elif cur_rad2 <= 0:
            cur_rad = cur_rad1
        else:
            cur_rad = min(cur_rad1, cur_rad2)

        self.annotationRadiusBuf += [cur_rad]
        if len(self.annotationRadiusBuf) > self.annotationBufferSize:
            self.annotationRadiusBuf = self.annotationRadiusBuf[1:]

        if (self.annotationDelayCnt == 0) or (self.annotationRadius <= 0):
            buff = np.array(self.annotationRadiusBuf, dtype = np.float64)
            buff = buff[buff > 0]

            if len(buff) > 0:
                self.annotationRadius = np.mean(buff)
            else:
                self.annotationRadius = 0

        # Calculate road center shift
        lane_btm_y = top_shape[0] * world_correction[0]
        if self.laneLines[0].isLineDetected and self.laneLines[1].isLineDetected:
            lane_btm_x1 = self.laneLines[0].lineShapeWorld[0] * (lane_btm_y ** 2) + self.laneLines[0].lineShapeWorld[1] * lane_btm_y + self.laneLines[0].lineShapeWorld[2]
            lane_btm_x2 = self.laneLines[1].lineShapeWorld[0] * (lane_btm_y ** 2) + self.laneLines[1].lineShapeWorld[1] * lane_btm_y + self.laneLines[1].lineShapeWorld[2]
            lane_size = lane_btm_x2 - lane_btm_x1
        else:
            lane_size = -1

        if lane_size > 0:
            lane_btm_y = top_shape[0]
            lane_btm_x1 = self.laneLines[0].lineShape[0] * (lane_btm_y ** 2) + self.laneLines[0].lineShape[1] * lane_btm_y + self.laneLines[0].lineShape[2]
            lane_btm_x2 = self.laneLines[1].lineShape[0] * (lane_btm_y ** 2) + self.laneLines[1].lineShape[1] * lane_btm_y + self.laneLines[1].lineShape[2]
            lane_image_points = self.camera.perspectiveTransformPoints(np.array([[lane_btm_x1, lane_btm_y], [lane_btm_x2, lane_btm_y]], dtype = np.float64), True)
            x1 = lane_image_points[0, 0]
            x2 = lane_image_points[1, 0]

            if x1 != x2:
                cur_shift = (img_shape[1] / 2.0 - (x1 + x2) / 2.0) / np.absolute(x2 - x1) * lane_size
            else:
                cur_shift = -10000000
        else:
            cur_shift = -10000000

        self.annotationCenterShiftBuf += [cur_shift]
        if len(self.annotationCenterShiftBuf) > self.annotationBufferSize:
            self.annotationCenterShiftBuf = self.annotationCenterShiftBuf[1:]

        if (self.annotationDelayCnt == 0) or (self.annotationCenterShift <= -1000000):
            buff = np.array(self.annotationCenterShiftBuf, dtype = np.float64)
            buff = buff[buff > -1000000]

            if len(buff) > 0:
                self.annotationCenterShift = np.mean(buff)
            else:
                self.annotationCenterShift = 0

        # Do vizualuzation
        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorBGR

        points = np.append(points1, points2[::-1], axis = 0)

        cv2.fillPoly(image_mask, [points], (255, 255, 255))

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorLineLeftBGR

        cv2.polylines(image_mask, [points1], 0, (255, 255, 255), thickness = 10)

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorLineRightBGR

        cv2.polylines(image_mask, [points2], 0, (255, 255, 255), thickness = 10)

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        annotationRadiusMul = max(1, int(np.round(self.annotationRadius / 100.0)))
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        cv2.putText(
            res_img,
            'Radius of curvature = {:.1f} km'.format(annotationRadiusMul / 10.0),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            self.annotationColorTextBGR,
            thickness = 2,
            lineType = cv2.LINE_AA)

        annotationCenterShiftMult = int(self.annotationCenterShift * 10.0)
        if annotationCenterShiftMult > 0:
            txt = 'Vehicle is {:.1f} m right of center'.format(annotationCenterShiftMult / 10.0)
        elif annotationCenterShiftMult < 0:
            txt = 'Vehicle is {:.1f} m left of center'.format(-annotationCenterShiftMult / 10.0)
        else:
            txt = 'Vehicle is by center'

        cv2.putText(
            res_img,
            txt,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            self.annotationColorTextBGR,
            thickness = 2,
            lineType = cv2.LINE_AA)

        return True, res_img

    def processFrame(
        self,
        img # Source image in OpenCV BGR format
        ):
        """
        Frame processing entry point
        """

        if self.visualization:
            self.visOrigImage = img.copy()

        # Do image undistortion
        img_undist = self.camera.undistort(img)

        if self.visualization:
            self.visUndistortImage = img_undist.copy()

        # Detect edges and do image binary
        img_binary = self.camera.makeBinary(img_undist)

        if self.visualization:
            self.visBinaryImage = self.camera.binaryToImg(img_binary)

        # Perform perspective transformation
        top_bin = self.camera.perspectiveTransformBinary(img_binary);

        if self.visualization:
            self.visTopViewBinaryImage = self.camera.binaryToImg(top_bin)

        # Detect lane lines
        self._detectLines(top_bin)

        # Calculate parameters and annotate image
        self.isImageAnnotated, self.visImageAnnotated = self._calculateParameters(img_undist, top_bin.shape)

        return self.visImageAnnotated




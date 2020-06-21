import numpy as np
import cv2
import os
from imageio import imread
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


class Frames:
    data_path = "./paths.txt"
    video_data_path = "video_data.npy"

    def __init__(self):
        self.paths = []
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as data_file:
                for l in data_file.readlines():
                    l = l.strip()
                    if len(l) != 0:
                        self.paths.append(l)

    def motion_compution(self, frames):
        """
        The method compute the motion between each frame by finding feature points and using lucas-kanade.
        :param frames: list of frames
        :return: list of dx,dy,angle motion per each frame.
        """
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        old_frame = frames[0]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        motions = []
        for i in range(1, len(frames)):
            p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.01,
                                         minDistance=7, blockSize=7)
            frame = frames[i]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st == 1]
            if len(good_new) == 0:
                break
            good_old = p0[st == 1]

            m, inliers = cv2.estimateAffine2D(good_old, good_new)
            dx = m[0, 2]
            dy = m[1, 2]
            phi = np.arctan2(m[1, 0], m[0, 0])
            motions.append([dx, dy, phi])

            old_gray = frame_gray
        return motions


    def warp_channel(self, frame, motion):
        """
        Warps a 2D image with a given homography.
        :param frame: a 2D image.
        :param h: homograhpy.
        :return: A 2d warped image.
        """
        dx, dy, phi = motion
        h = np.zeros((2, 3))
        h[0, 0] = np.cos(phi)
        h[0, 1] = -np.sin(phi)
        h[1, 0] = np.sin(phi)
        h[1, 1] = np.cos(phi)
        h[0, 2] = 0
        h[1, 2] = dy
        return cv2.warpAffine(frame, h, (frame.shape[1], frame.shape[0]))

    def apply_moving_average_filter(self, curve, radius):
        """
        moving average filter
        :param curve: unsmooth curve
        :param radius:radius
        :return: smooth curve
        """
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]
        return curve_smoothed


    def smooth_transform(self, transforms):
        """
        return a smooth transformations after calculate a smooth trajectory using moving average filter
        :param transforms: transform to be smooth
        :return:
        """
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = self.apply_moving_average_filter(trajectory[:, i], radius=2)
        diff = smoothed_trajectory - trajectory
        return transforms + diff

    def fix_borders(self, frame):
        """
        The method fix the border of a given frame according to the rotation
        :param frame: given frame
        :return: rotate frame
        """
        s = frame.shape
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        return cv2.warpAffine(frame, T, (s[1], s[0]))

    def read_and_fix_images(self, path):
        """
        The method read all the frames, fix and transform it to be correct according to the first frame.
        Then calculate the motion between each frame.
        :param path: path to the frames
        :return: fixed frame and array of the motions
        """
        frames = [imread(os.path.join(path, filename)) for filename in os.listdir(path)
                  if not filename.endswith('npy')]
        smooth_motions = self.smooth_transform(self.motion_compution(frames))
        fixed_frames, motions = [], []

        for i in range(len(frames) - 1):
            fixed = self.warp_channel(frames[i], smooth_motions[i])
            fixed_frames.append(self.fix_borders(fixed))
            motions.append(int(np.ceil(abs(smooth_motions[i][0]))))

        fixed_frames.append(self.fix_borders(frames[-1]))

        video_data_path = os.path.join(path, self.video_data_path)
        np.save(video_data_path, np.array([fixed_frames, np.array(motions)]))
        return fixed_frames, np.array(motions)

    def get_extra_col(self, motion_per_frame, first_col, last_col):
        """
        The method calculate the extra columns that need to be taken from each frame according to the motion
        :param motion_per_frame: motion to calculate according to
        :param first_col: col to start with
        :param last_col: col to end with
        :return: 1d array of the amount of extra columns to add to each slit(can be negative)
        """
        if first_col <= last_col:
            extra_motion = ((last_col - first_col) * motion_per_frame / sum(motion_per_frame)).astype(np.int)
        else:
            extra_motion = -((first_col - last_col) * motion_per_frame / sum(motion_per_frame)).astype(np.int)
        return extra_motion

    def change_view_point(self, frames, motions, first_frame, first_col, last_frame, last_col):
        """
        The method change the view point if the image, depending on the given columns and frame to start and end with.
        :param frames: list of frames from left to right sequence
        :param motions: list of motion in the x space
        :param first_frame: frame from sequence
        :param first_col: first column from first frame
        :param last_frame:  frame from sequence
        :param last_col:last column from last frame
        :return: new image with changed view point
        """
        if first_frame == last_frame:
            return frames[first_frame][:, first_col:last_col + 1].astype(np.uint8)
        motion_per_frame = motions[first_frame: last_frame + 1].copy()

        extra_col = self.get_extra_col(motion_per_frame, first_col, last_col)
        width = sum(motion_per_frame + extra_col)
        height = frames[0].shape[0]
        if width < 0:
            return None
        im = np.zeros((height, 0, 3))

        cur_col = 0
        for i in range(len(motion_per_frame)):
            cols_amount = motion_per_frame[i] + extra_col[i]
            frame_strip = frames[first_frame + i][:, first_col:first_col + cols_amount]

            im = np.hstack((im, frame_strip))

            cur_col += cols_amount
            first_col += extra_col[i]

        return im.astype(np.uint8)

    def image_averaging(self, im, frames, left_to_mid_motion, right_to_mid_motion, movement):
        """
        The method cut frames according to the motion and and then average them in the right position in
        the middle frame, and with this creating a image that refocus on different place in the image(depending on
        the movement).
        :param im: image to start calculation with(middle frame)
        :param frames: list of frames from left to right sequence
        :param left_to_mid_motion: list of motion from the left frame to middle
        :param right_to_mid_motion: list of motion from the right frame to middle
        :param movement:
        :return: refocus image
        """
        mid = len(frames) // 2
        width = im.shape[1]
        sum_per_col = np.ones(width)
        for i in range(len(left_to_mid_motion)):
            motion = left_to_mid_motion[i]
            if motion >= width:
                return None
            if movement < 0:
                cur_frame = mid + 1 + i
            else:
                cur_frame = i
            im[:, :width - motion] += frames[cur_frame][:, motion:]
            sum_per_col[:width - motion] += 1
        for i in range(len(right_to_mid_motion)):
            motion = right_to_mid_motion[i]
            if motion >= width:
                return None
            if movement < 0:
                cur_frame = i
            else:
                cur_frame = mid + 1 + i
            im[:, motion:] += frames[cur_frame][:, :width - motion]
            sum_per_col[motion:] += 1
        return (im / sum_per_col[:, np.newaxis]).astype(np.uint8)

    def refocusing(self, frames, motions, movement):
        """

        :param frames: list of frames from left to right sequence
        :param motions: list of motion in the x space between each frame
        :param movement:
        :return: refocus image
        """
        mid = len(frames) // 2
        im = frames[mid].copy().astype(np.int)
        height, width = im.shape[0], im.shape[1]
        factor = np.abs(movement / width)

        left_to_mid_motion = np.round(np.cumsum(motions[:mid][::-1])[::-1] * factor).astype(np.int)
        right_to_mid_motion = np.round(np.cumsum(motions[mid:]) * factor).astype(np.int)
        if movement < 0:  # change to negative side
            left_to_mid_motion, right_to_mid_motion = right_to_mid_motion, left_to_mid_motion

        return self.image_averaging(im, frames, left_to_mid_motion, right_to_mid_motion, movement)



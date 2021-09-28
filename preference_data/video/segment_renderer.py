from abc import ABC, abstractmethod
import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
from pathlib import Path


class AbstractSegmentRenderer(ABC):
    def __init__(self, env, out):
        self.env = env
        self.out = out

    @abstractmethod
    def render_segment(self, segment):
        pass


class SegmentRenderer(AbstractSegmentRenderer):

    def __init__(self, env, out):
        super().__init__(env, out)
        self.nr = 1

    def render_segment(self, segment):
        outfile = self.out + str(self.nr) + '.avi'
        fps = 8
        fourcc = VideoWriter_fourcc(*'DIVX')
        singleframe = np.array(segment[0].observation)
        fshape = singleframe.shape[1:3]
        vid_writer = VideoWriter(outfile, fourcc, fps, fshape, False)

        for frame in segment:
            framestack = np.array(frame.observation)
            writeframe = cv2.normalize(
                src=framestack[0], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8S)
            vid_writer.write(writeframe)
        vid_writer.release()
        self.nr = self.nr + 1

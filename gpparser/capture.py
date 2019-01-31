import sys
import cv2


def get_frame_size(cap):
    return tuple((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


class GazePointCapture(cv2.VideoCapture):

    def __init__(self, filename):
        super().__init__(filename)
        self.frame_shape = get_frame_size(self)
        self.total_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_pos(self):
        """
        Returns current frame position.
        """
        return self.get(cv2.CAP_PROP_POS_FRAMES)

    def set_frame_pos(self, key):
        """
        Set current frame position.

        Parameters
        ----------
        key : int
            Frame position.
        """
        assert isinstance(key, int), type(key)
        assert abs(key) <= self.total_frames
        try:
            if key < 0:
                key = self.total_frames + key
            self.set(cv2.CAP_PROP_POS_FRAMES, key)
            return True
        except Exception as error:
            print(error, file=sys.stderr)
            return False

    def get_sec_pos(self):
        """
        Returns current time position in seconds.
        """
        return self.get(cv2.CAP_PROP_POS_MSEC) * 1e3

    def set_sec_pos(self, sec):
        try:
            self.set(cv2.CAP_PROP_POS_MSEC, sec * 1e-3)

        except Exception as error:
            print(error, file=sys.stderr)

            return False

    def get_frame(self, key):
        """
        Read specific frame number.

        Parameters
        ----------
        key : int
            Frame number.

        Returns
        -------
        frame : np.ndarray
        """
        if not self.isOpened():
            self.open()

        self.set_frame_pos(key)

        return self.read()[1]

    def __iter__(self):
        while self.isOpened():
            yield self.read()[1]


def test():
    import sys

    filename = sys.argv[1]
    gp_cap = GazePointCapture(filename)

    print(dir(gp_cap))
    print(gp_cap.frame_shape)
    print(gp_cap.total_frames)

    frame_index = -100
    test_frame = gp_cap.get_frame(frame_index)

    window_title = f'test_frame no.{frame_index}'
    cv2.imshow(window_title, cv2.resize(test_frame, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(3000)
    cv2.destroyWindow(window_title)

    gp_cap.set_frame_pos(0)

    # for frame in gp_cap:
    #     cv2.imshow('cap', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    iter_frames = iter(gp_cap)

    for i in range(100):
        test_frame = next(iter_frames)
        cv2.imshow(window_title,
                   cv2.resize(test_frame, (0, 0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()

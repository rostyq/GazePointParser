import gzip
import ast
import re

from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np
import cv2

from tqdm import tqdm

from fixation import Fixation


def parse_records(filename, custom_record=None, encoding='utf-8'):

    data_match = re.compile(bytes(r'Data:\r?\n', encoding), re.MULTILINE)
    filter_match = re.compile(bytes(r'\s|\r?\n', encoding))
    record_repl = bytes(r'\1"\2":', encoding)
    raw_repl = b''
    split_match = re.compile(bytes(r'}?-{', encoding))
    format_match = re.compile(bytes(r'(,?)(\w+):', encoding))

    with gzip.open(filename, 'rb') as file:
        raw = file.read()
        result = data_match.search(raw)

        if custom_record is not None:
            assert type(custom_record)
            # keys = custom_record._fields

        if result is not None:
            data_start_position = result.end() + 1
            data = filter_match.sub(raw_repl, raw[data_start_position:])

            for raw_record in split_match.split(data)[1:-1]:
                formatted_record = format_match.sub(record_repl, raw_record)
                yield ast.literal_eval((b'{%s}' % formatted_record).decode())
        else:
            raise Exception


def get_gaze(record, frame_size):
    width, height = frame_size
    return int(record['FPOGX'] * width), int((record['FPOGY']) * height)


def get_frame_size(cap):
    return tuple((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


class ProjectSession:

    YML_FILE = {
        'name': 'user',
        'enc': 'utf-8',
        'ext': '.yml.gz',
    }

    SCRN_FILE = {
        'name': 'scrn',
        'ext': '.avi',
    }

    SCRN_EXP_FILE = {
        'name': 'screen',
        'ext': '.mp4',
    }

    CAM_FILE = {
        'name': 'cam',
        'ext': '.avi',
    }

    CAM_EXP_FILE = {
        'name': 'face',
        'ext': 'mp4',
    }

    ANNOT_FILE = {
        'name': 'screen',
        'ext': '.txt',
        'names': ['timestamp', 'flag'],
    }

    GAZE_FILE = {
        'name': 'gazes',
        'ext': '.csv',
    }

    FIX_FILE = {
        'name': 'fixations',
        'ext': '.csv',
    }

    FOURCC = 0x7634706d

    USER_DIR = 'user'
    SRC_DIR = 'src'
    RESULT_DIR = 'result'

    DIR_FMT = '{name}_{index}'
    RAW_FMT = '{index:0>4}-{name}{ext}'
    EXP_FMT = '{name}{ext}'

    def __init__(self, index, name, prj_path,
                 records_range, info=None, custom_record=None):
        assert isinstance(index, int)
        assert isinstance(name, str)
        assert prj_path.is_dir()

        if info is not None:
            assert isinstance(info, dict)
            self.color = tuple(map(int, info.get('Color')[:-1]))
        else:
            self.color = (255, 0, 0)

        if custom_record is not None:
            assert callable(custom_record)

        if records_range is not None:
            assert isinstance(records_range, (tuple, int))

        self.index = index
        self.name = name
        self.info = info

        # define path`s
        self.user_path = prj_path / self.USER_DIR
        self.src_path = prj_path / self.SRC_DIR
        self.result_path = prj_path / self.RESULT_DIR
        self.export_path = self._get_export_dir_path()

        # paths to raw data
        self.yml_path = self._get_raw_filename(self.YML_FILE)
        self.scrn_path = self._get_raw_filename(self.SCRN_FILE)
        self.cam_path = self._get_raw_filename(self.CAM_FILE)

        # path to exported files
        self.screen_path = self._get_exp_filename(self.SCRN_EXP_FILE)
        self.gaze_path = self._get_exp_filename(self.GAZE_FILE)
        self.fixation_path = self._get_exp_filename(self.FIX_FILE)
        self.annot_path = self._get_exp_filename(self.ANNOT_FILE)

        self.records_range = range(*records_range)
        self.custom_record = custom_record
        self.records = None

    @classmethod
    def from_prj_entry(cls, info, path):
        """
        Creates ProjectSession instance from data
        written in %project_name%.prj.

        Parameters
        ----------
        info : dict
            Dictionary containing data about session.
        prj_path : str or pathlib.Path
            Path to project dir dir in project directory.
        """
        assert isinstance(info, dict)
        index = info.get('Id')
        name = info.get('Name')
        records_range = (info.get('DataRecords'),)
        return cls(index=index, name=name, info=info,
                   prj_path=path, records_range=records_range)

    def _get_export_dir_path(self):
        return self.result_path / self.DIR_FMT.format(index=self.index,
                                                      name=self.name)

    def _get_raw_filename(self, file_conf):
        return self.user_path / self.RAW_FMT.format(index=self.index,
                                                    **file_conf)

    def _get_exp_filename(self, file_conf):
        return self.export_path / self.EXP_FMT.format(**file_conf)

    def iter_records(self):
        """
        Return iterator on records.
        """
        return parse_records(self.yml_path,
                             custom_record=self.custom_record,
                             encoding=self.YML_FILE['enc'])

    def get_dataframe(self):
        """
        Returns copy of session data.

        Returns:
        --------
            dataframe : pd.DataFrame
        """
        if self.records is None:
            self.records = pd.DataFrame(self.iter_records())
        return self.records.copy()

    def export_cam(self, path):
        """
        doc
        """
        # TODO write cam video export
        raise NotImplementedError

    def render_screen(self, filename, last_fixation_count=5, verbose=True):
        """
        Render screen gaze video and save as mp4 file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path where to save video file.
        last_fixation_count : int
            Defines how many fixations render on each screen frame.
        """
        scrn_cap = cv2.VideoCapture(str(self.scrn_path))
        df = self.get_dataframe()

        frame_size = get_frame_size(scrn_cap)
        scrn_writer = cv2.VideoWriter(str(filename), self.FOURCC, fps=60,
                                      frameSize=frame_size)

        # create flags which indicate that frame has been changed
        df[['NEW_SCN', 'NEW_FID']] = df[['SCN', 'FPOGID']] \
            .diff().fillna(True).astype(bool)

        # convert relative gaze coordinate to screen pixel coordinates
        df[['PIX_FPOGX', 'PIX_FPOGY']] = (
            df[['FPOGX', 'FPOGY']] * np.array(frame_size)
        ).astype(int)

        # init fixation
        current_fixation = Fixation.from_record(df.iloc[0])

        # create fixation deque for visualizing last `last_fixation_count`
        fixations_deque = deque([current_fixation], maxlen=last_fixation_count)

        with tqdm(total=len(self.records)) as pbar:
            for i, record in df.iterrows():
                try:
                    if verbose:
                        pbar.update(1)

                    if record['NEW_SCN']:
                        scrn_frame = scrn_cap.read()[1]

                    if scrn_frame is not None:
                        scrn_frame_copy = np.copy(scrn_frame)

                    if record['NEW_FID']:
                        fixations_deque.append(current_fixation)
                        current_fixation = Fixation.from_record(record)
                    else:
                        current_fixation.update(record)

                    for i, fixation in enumerate(fixations_deque):
                        if i > 0:
                            fixation.trace(scrn_frame_copy,
                                           self.color,
                                           fixations_deque[i-1])
                    current_fixation.trace(scrn_frame_copy,
                                           self.color,
                                           fixations_deque[-1])

                    scrn_writer.write(scrn_frame_copy)

                    if verbose:
                        cv2.imshow('SCRN', cv2.resize(scrn_frame_copy, (0, 0),
                                                      fx=0.5, fy=0.5))
                        key = cv2.waitKey(1)
                        if key == 27:
                            break

                except StopIteration:
                    scrn_frame = scrn_frame_copy

        scrn_writer.release()

    def export_all_gaze(self, filename, **kwargs):
        """
        Exports all session gaze records to file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to save.
        **kwargs : dict
            Additional keyword arguments, that will be passed to
            pandas.DataFrame.to_csv() method.
        """
        self.records.to_csv(str(filename), **kwargs)

    def export_fixations(self, filename, **kwargs):
        """
        Exports only gaze fixation from session gaze records to `path`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to save.
        **kwargs : dict
            Additional keyword arguments, that will be passed to
            pandas.DataFrame.to_csv() method.
        """
        # TODO implement fixation aggregation and exporting
        raise NotImplementedError

    def read_annotation_file(self, filename):
        """
        Read screen video annotation file.

        Parameters
        ----------
        filename : str or pathlib.Path
            File name of annotation plain-text file.
        """
        return pd.read_csv(filename, sep=',', header=None,
                           names=self.ANNOT_FILE['names'])

    def split_records(self):
        """
        Splits data records into flag chunks from
        annotation file.
        """
        # TODO continue
        raise NotImplementedError
        annot = self.read_annotation_file(self.annot_path)
        records = self.get_dataframe()
        
        records['CHUNK_NUM'] = 0
        records['LABEL'] = 0

        for i, row in annot.iterrows():
            timestamp = row['timestamp']
            flag = row['flag']

        print(annot_df)

    def export(self, screen=True, gaze=True, fixation=False,
               last_fixation_count=5):
        """
        Export data to `path` dir.
        Results will be saved in (project_directory)/result/.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to destination dir.
        screen : bool
            render video
        gaze : bool
            export all gaze
        fixation : bool
            export fixation data
        last_fixation_count : int
            how many fixations to render on each frame.
        """

        # create dir for export
        if not self.export_path.exists():
            self.export_path.mkdir()

        print(f'Start export {self.index} {self.name} session.')
        if screen:
            print(f'Start rendering video:')
            self.render_screen(
                               last_fixation_count,
                               verbose=True)
        if gaze:
            print(f'Export all gaze.')
            self.export_all_gaze(self.gaze_path)
        if fixation:
            print(f'Export fixations.')
            self.export_fixations(self.fixation_path)


def test():
    import sys

    # define test paths with GP Analysis data
    test_sess_path = Path(sys.argv[1])
    records = int(sys.argv[2])

    sess = ProjectSession(index=7,
                          name='Саша',
                          prj_path=Path(test_sess_path),
                          records_range=(records,))
    # sess.render_screen(test_export_path)
    sess.split_records()


if __name__ == '__main__':
    test()

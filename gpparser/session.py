import gzip
import ast
import re

from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT

from pathlib import Path
from collections import deque
from collections import namedtuple

import pandas as pd
import numpy as np
import cv2

from tqdm import tqdm

from gpparser.fixation import Fixation
from gpparser.capture import GazePointCapture


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
        'names': ['fixation'],
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
    MPV_CMD_FMT = 'mpv {path} --pause=yes --title="{session} screen.mp4 - mpv"'

    SESS_RECORD_COLS = ['index', 'name', 'records',
                        'rendered', 'annotated', 'splitted']
    SessionInfo = namedtuple('SessionInfo', SESS_RECORD_COLS)

    def __init__(self, index, name, prj_path,
                 records_range, raw_info=None, custom_record=None):
        assert isinstance(index, int)
        assert isinstance(name, str)
        assert prj_path.is_dir()

        if raw_info is not None:
            assert isinstance(raw_info, dict)
            self.color = tuple(map(int, raw_info.get('Color')[:-1]))
        else:
            self.color = (255, 0, 0)

        if custom_record is not None:
            assert callable(custom_record)

        if records_range is not None:
            assert isinstance(records_range, (tuple, int))

        self.index = index
        self.name = name
        self.raw_info = raw_info if raw_info is not None else {}

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

        # splitted path
        self.split_path = self.export_path / 'splitted'

        # records
        self.records_range = range(*records_range)
        self.custom_record = custom_record
        self.records = None

        # captures
        self.scrn_cap = GazePointCapture(str(self.scrn_path))
        self.cam_cap = GazePointCapture(str(self.cam_path))

    def __str__(self):
        return f'{self.index}:{self.name}'

    @classmethod
    def from_prj_entry(cls, raw_info, path):
        """
        Creates ProjectSession instance from data
        written in %project_name%.prj.

        Parameters
        ----------
        raw_info : dict
            Dictionary containing data about session.
        prj_path : str or pathlib.Path
            Path to project dir dir in project directory.
        """
        assert isinstance(raw_info, dict)
        index = raw_info.get('Id')
        name = raw_info.get('Name')
        records_range = (raw_info.get('DataRecords'),)
        return cls(index=index, name=name, raw_info=raw_info,
                   prj_path=path, records_range=records_range)

    def _get_export_dir_path(self):
        return self.result_path / self.DIR_FMT.format(index=self.index,
                                                      name=self.name)

    def _get_raw_filename(self, file_conf):
        return self.user_path / self.RAW_FMT.format(index=self.index,
                                                    **file_conf)

    def _get_exp_filename(self, file_conf):
        return self.export_path / self.EXP_FMT.format(**file_conf)

    def get_session_info(self):
        return self.SessionInfo(
            index=self.index,
            name=self.name,
            records=self.raw_info.get('DataRecords', len(self.records_range)),
            rendered=self.screen_path.is_file(),
            annotated=self.annot_path.is_file(),
            splitted=self.split_path.is_dir(),
            )

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

    def render_screen(self, filename, last_fixation_count=5, verbose=False):
        """
        Render screen gaze video and save as mp4 file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path where to save video file.
        last_fixation_count : int
            Defines how many fixations render on each screen frame.
        """
        df = self.get_dataframe()

        scrn_writer = cv2.VideoWriter(str(filename), self.FOURCC, fps=60,
                                      frameSize=self.scrn_cap.frame_shape)

        # create flags which indicate that frame has been changed
        df[['NEW_SCN', 'NEW_FID']] = df[['SCN', 'FPOGID']] \
            .diff().fillna(True).astype(bool)

        # convert relative gaze coordinate to screen pixel coordinates
        df[['PIX_FPOGX', 'PIX_FPOGY']] = (
            df[['FPOGX', 'FPOGY']] * np.array(self.scrn_cap.frame_shape)
        ).astype(int)

        # init fixation
        current_fixation = Fixation.from_record(df.iloc[0])

        # create fixation deque for visualizing last `last_fixation_count`
        fixations_deque = deque([current_fixation], maxlen=last_fixation_count)

        # create frames iterator
        iter_scrn_frames = iter(self.scrn_cap)

        with tqdm(total=len(self.records)) as pbar:
            for i, record in df.iterrows():
                try:
                    pbar.update(1)

                    if record['NEW_SCN']:
                        scrn_frame = next(iter_scrn_frames)

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
        self.get_dataframe().to_csv(str(filename), **kwargs)

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
        records = self.get_dataframe()
        records.groupby(by='FPOGID').tail(1).to_csv(str(filename), **kwargs)

    def read_annotation_file(self, filename):
        """
        Read screen video annotation file.

        Parameters
        ----------
        filename : str or pathlib.Path
            File name of annotation plain-text file.
        """
        return pd.read_csv(filename, sep=',', header=None, comment='#',
                           names=self.ANNOT_FILE['names'])

    def match_annotation(self, shift=1):
        """
        Match every data record to flag and chunk number from
        annotation file. Returns modificated records
        dataframe (which comes from `get_dataframe()`).

        Returns
        -------
            records: pd.DataFrame
        """
        annot = self.read_annotation_file(self.annot_path)
        records = self.get_dataframe()

        chunk_indices = np.full(len(records), np.nan)

        prev_fixation = 0
        for chunk_id, row in annot.iterrows():

            next_fixation = row['fixation']
            mask = (records['FPOGID'] > prev_fixation) & \
                   (records['FPOGID'] <= next_fixation)

            chunk_indices[mask] = chunk_id
            prev_fixation = next_fixation

        mask = records['FPOGID'] > next_fixation
        chunk_indices[mask] = chunk_id + 1

        records['CHUNK'] = chunk_indices.astype(int)
        records['LABEL'] = (chunk_indices % 2 + shift).astype(int)
        return records

    def split(self, dir_names=None):
        """
        Split into chunks records and cam videos rely on
        flags written in annotation plain text
        txt file.

        Saves chunks to corresponding
        `splitted/label_{flag}/` directory.

        Parameters
        ----------
        dir_names : dict[int:str]
            Mapping for label directory names. Keys stand for
            labels, values stand for names.
        """
        try:
            print(f'Processing split for session {self}')
            matched = self.match_annotation()

            # create flags which indicate that frame has been changed
            matched['NEW_CAM'] = matched['CAM'] \
                .diff().fillna(True).astype(bool)

            # check dir
            if not self.split_path.exists():
                self.split_path.mkdir()

            # create dir labels
            labels = np.unique(matched['LABEL'])
            if dir_names is None:
                dir_names = {label: f'label_{label}' for label in labels}

            # create dirs for labels
            label_dirs = {}
            for label in labels:
                label_dir_path = self.split_path / dir_names[label]
                if not label_dir_path.exists():
                    label_dir_path.mkdir()
                label_dirs[label] = label_dir_path

            # create frames iterator
            iter_cam_frames = iter(self.cam_cap)

            for i, chunk in matched.groupby(by='CHUNK'):
                # get chunk identifications
                chunk_name = f'{i:0>8}'
                label = chunk['LABEL'].iloc[0]

                chunk_path = label_dirs[label] / chunk_name
                cam_path = str(chunk_path.with_suffix('.mp4'))
                chunk_cam_writer = cv2.VideoWriter(
                        cam_path,
                        self.FOURCC,
                        fps=30,
                        frameSize=self.cam_cap.frame_shape
                        )

                cam_frame = None

                print(f'Processing chunk no. {i}')
                with tqdm(total=len(chunk)) as pbar:
                    for j, row in chunk.iterrows():
                        pbar.update(1)

                        if row['NEW_CAM']:
                            cam_frame = next(iter_cam_frames)
                            chunk_cam_writer.write(cam_frame)

                chunk_cam_writer.release()
                cv2.destroyAllWindows()
                chunk.to_csv(chunk_path.with_suffix('.csv'))
        except FileNotFoundError:
            print(f'Annotation for {self} in {self.annot_path} not found.')

    def export(self, screen=True, gaze=True, fixation=False,
               last_fixation_count=5, verbose=False):
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
            self.render_screen(self.screen_path,
                               last_fixation_count,
                               verbose=verbose)
        if gaze:
            print(f'Export all gaze.')
            self.export_all_gaze(self.gaze_path)
        if fixation:
            print(f'Export fixations.')
            self.export_fixations(self.fixation_path)

    def annotate(self):
        """
        Opens video `screen.mp4 from session directory and writes
        users input to `screen.txt`.
        """
        if self.screen_path.is_file():

            if self.annot_path.exists():
                print(f'Session {self} has annotation file. Overwrite?')
                answer = input('(y, yes -- overwrite, n, no -- pass)\n> ')
                if answer.lower() in ['n', 'no']:
                    return
                elif answer.lower() in ['y', 'yes']:
                    pass
                elif answer == 'exit':
                    exit()

            # open screen video in mpv
            command = self.MPV_CMD_FMT.format(
                    path=str(self.screen_path),
                    session=str(self)
                    )
            process = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT)

            fixations = []
            print('Input fixation indices:')
            print('`end` -- to end this session, `exit` -- end procedure')
            while 'Annotation Loop':
                answer = input('> ')
                if answer in ['exit', 'end']:
                    process.kill()
                    break
                elif answer == 'rm':
                    try:
                        removed = fixations.pop()
                        print(f'Removed {removed}')
                    except IndexError:
                        print('Fixation list is empty.')
                elif answer == 'ls':
                    print(fixations)
                else:
                    try:
                        fixation = int(answer)
                        if fixations:
                            assert fixation > fixations[-1]
                        fixations.append(fixation)
                    except ValueError:
                        print('Wrong input')
                        continue
                    except AssertionError:
                        print('New fixation must be bigger than previous.')
                        continue

            with self.annot_path.open('w') as annot_file:
                annot_file.write('# Annotated fixation indices\n')
                payload = '\n'.join(map(str, fixations))
                annot_file.write(payload)

            if answer == 'exit':
                process.kill()
                exit()

            process.wait()
        else:
            print(f'Session {self} has no screen video.')


def test():
    import sys

    # define test paths with GP Analysis data
    test_sess_path = Path(sys.argv[1])
    records = int(sys.argv[2])

    sess = ProjectSession(index=6,
                          name='Саша',
                          prj_path=Path(test_sess_path),
                          records_range=(records,))
    print(sess.get_session_info())


if __name__ == '__main__':
    test()

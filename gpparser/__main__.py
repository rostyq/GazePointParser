#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_argument_parser():

    from argparse import ArgumentParser

    description = """
    GazePoint Analysis project reader, exporter
    and preprocessor written in Python.
    """

    project_dir_help = """
    Path to GazePoint project dir.
    If no another parameters was specified
    print project sessions information table.
    """

    export_help = """
        Export data from raw `user` dir to `result` dir.
        Can be specified with -s, -f, -g flags.
        If no flag was specified, will ask further.
    """

    split_help = """
        Read video annotation data in `results`
        and splits exported data into flag-chunks
        relying on specified in annotation data with timestamps.
    """

    annotate_help = """
        Run annotation video. Opens in mpv screen video from session
        and in console waits for fixation-stamps. Recieved fixation
        indices will be written in /result/session_name/screen.txt.
    """

    w_help = """
        How much fixations need to be rendered on each screen frame. Default 5.
    """

    a_help = """
        Process all session in project.
    """

    r_help = """
        Toggle screen rendering.
    """

    f_help = """
        Toggle fixation data exporting.
    """

    g_help = """
        Toggle all gaze data exporting.
    """

    i_help = """
        Shows image during rendering.
    """

    def bool_ans(ans):
        if ans.lower() in ['y', 'yes']:
            return True
        elif ans.lower() in ['n', 'no']:
            return False
        else:
            print('Ununderstood answer. Close.')
            exit()

    def cq(msg):
        return f'{msg} (yes, no)\n> '

    parser = ArgumentParser(description=description)
    parser.add_argument('project_dir',
                        type=str,
                        help=project_dir_help)

    parser.add_argument('--export', action='store_true', help=export_help)
    parser.add_argument('--split', action='store_true', help=split_help)
    parser.add_argument('--annotate', action='store_true', help=annotate_help)

    parser.add_argument('--sessions',
                        metavar='N', type=int, default=None, nargs='+',
                        help='Session indices which should be processed.')
    parser.add_argument('--all', action='store_true', help=a_help)

    parser.add_argument('-r', '--render', action='store_true', help=r_help)
    parser.add_argument('-w', '--window', type=int, default=5, help=w_help)

    parser.add_argument('-f', '--fixations', action='store_true', help=f_help)
    parser.add_argument('-g', '--gazes', action='store_true', help=g_help)
    parser.add_argument('-i', '--imshow', action='store_true', help=i_help)

    return parser


def main(args=None):
    """
    Main function.
    """
    if args is None:
        args = get_argument_parser().parse_args()

    from pathlib import Path
    from gpparser.project import GazePointProject

    project = GazePointProject(Path(args.project_dir))

    session_indices = None if args.all else args.sessions

    print(f"Information about available sessions in {project.path}:")
    print(project.get_sessions_info().replace(
          {True: '+', False: '-'}
          ).to_string())

    if args.export:
        project.export_sessions(session_indices,
                                screen=args.render,
                                fixation=args.fixations,
                                gaze=args.gazes,
                                verbose=args.imshow,
                                last_fixation_count=args.window)

    if args.annotate:
        project.annotate_sessions(session_indices)

    if args.split:
        project.split_sessions(session_indices)


if __name__ == '__main__':
    main()

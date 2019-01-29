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

    parser.add_argument('--sessions',
                        metavar='N', type=int, default=None, nargs='+',
                        help='Session indices which should be processed.')
    parser.add_argument('-a', action='store_true', help=a_help)

    parser.add_argument('-r', action='store_true', help=r_help)
    parser.add_argument('-w', type=int, default=5, help=w_help)

    parser.add_argument('-f', action='store_true', help=f_help)
    parser.add_argument('-g', action='store_true', help=g_help)
    parser.add_argument('-i', action='store_true', help=i_help)

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

    print('Available sessions:')
    print('{:>5} {:<20} {:>10} {:9}'.format(
            'index',
            'name',
            'records',
            'annotated'
        )
    )
    for sess in project.sessions:
        index = sess.index
        name = sess.name
        annotated = '+' if sess.annot_path.is_file() else '-'
        records = sess.info.get('DataRecords')

        print(f'{index:>5} {name:<20} {records:>10} {annotated:>9}')

    session_indices = args.sessions if not args.a else args.sessions
    if args.export:
        project.export_sessions(session_indices,
                                screen=args.r,
                                fixation=args.f,
                                gaze=args.g,
                                verbose=args.i,
                                last_fixation_count=args.w)
    elif args.split:
        project.split_sessions(session_indices)


if __name__ == '__main__':
    main()

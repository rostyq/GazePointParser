#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from pathlib import Path


description = """
GazePoint Analysis project reader, exporter
and preprocessor written in Python.
"""

help_commands = """
Available commands:

export
    Export data from raw `user` dir to `result` dir.
    Can be specified with -s, -f, -g flags.
    If no flag was specified, will ask further.

split
    Read video annotation data in `results`
    and splits exported data into flag-chunks
    relying on specified in annotation data with timestamps.

preprocess
    Preprocess splitted data in `annotated` project directory
    into numpy arrays.
"""


def main():

    from argparse import ArgumentParser
    from argparse import ArgumentError


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
    parser.add_argument('command', type=str, help='Specifies what to be done. Type `list` to see available commands and their description.')
    parser.add_argument('-p', '--project_dir', type=str, help="Path to GazePoint project dir.")
    parser.add_argument('-s', '--sessions', metavar='N', type=int, default=None, nargs='+', help='Session indices which should be processed.')
    parser.add_argument('-r', action='store_true', help='Toggle screen rendering.')
    parser.add_argument('-f', action='store_true', help='Toggle fixation data exporting.')
    parser.add_argument('-g', action='store_true', help='Toggle all gaze data exporting.')
    parser.add_argument('-a', action='store_true', help='Process all session in project.')
    parser.add_argument('-w', type=int, default=5, help='How much fixations need to be rendered on each screen frame. Default 5.')

    args = parser.parse_args()

    if args.command == 'list':
        print(help_commands)
    elif args.command == 'export':
        from project import GazePointProject
        from project import ProjectSession

        command = args.command
        project_dir = Path(args.project_dir)

        project = GazePointProject(project_dir)
        
        print('Available sessions:')
        for sess in project.sessions:
            index = sess.index
            name = sess.name
            records = sess.info.get('DataRecords')

            print(f'{index:>3}\t{name:<25}\t{records:>10}')
        
        if not args.a:
            if not args.sessions:
                raw_sess_indices = input('Which sessions to export? (write comma separated sess indices)\n> ')
                if raw_sess_indices:
                    sess_indices = list(map(int, raw_sess_indices.split(',')))
                    print(f'Exporting: {sess_indices}.') 
                else:
                    sess_indices = None
                    print('Exporting all sessions.')
            else:
                sess_indices = args.sessions

        render_screen = args.r 
        export_fixation = args.f
        export_gaze = args.g
        last_fixation_count = args.w 

        project.export_sessions(sess_indices,
                                screen=render_screen,
                                fixation=export_fixation,
                                gaze=export_gaze,
                                last_fixation_count=last_fixation_count)
    else:
        print('Wrong command.')
        print(help_commands)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import time

HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'
CLEAR_TERM = '\033[H'

def blen(line):
    return len(re.sub('\033.*?m', '', line))

def quote(s):
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return s

def do(args):
    cmd = args.command
    if args.interactive_shell:
        cmd = f'bash -ic "{quote(cmd)}"'
    if args.pseudo_terminal:
        cmd = f'script -qec "{quote(cmd)}" /dev/null'
    while True:
        termsize = os.get_terminal_size()
        fillout = termsize.columns
        def pad(line):
            n = blen(line)
            over = n % fillout
            if not over and n:
                over = fillout
            return fillout - over
        linecount = 0

        capture = subprocess.run(cmd, shell=True, capture_output=True)
        lines = capture.stdout.decode().replace('\r\n', '\n').split('\n')
        lines += capture.stderr.decode().replace('\r\n', '\n').split('\n')
        lines = [line + ' ' * pad(line) for line in lines]
        for line in lines:
            assert blen(line) % fillout == 0
            if blen(line) > 0:
                assert blen(line) // fillout >= 1
            else:
                print(repr(line))
        linecount = sum(blen(line) // fillout for line in lines)
        print(CLEAR_TERM, end='')
        print('\n'.join(lines))
        for i in range(linecount, termsize.lines - 1):
            print(' ' * termsize.columns)
        time.sleep(args.interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--interval', metavar='seconds', action='store', type=int, default=5, help='seconds to wait between updates')
    parser.add_argument('-t', '--pseudo-terminal', action='store_true', help='run command using `script` to fake a TTY')
    parser.add_argument('-i', '--interactive-shell', action='store_true', help='run command using `bash -i`')
    parser.add_argument('command')
    args = parser.parse_args()
    print(HIDE_CURSOR, end='')
    try:
        do(args)
    finally:
        print(SHOW_CURSOR, end='', flush=True)

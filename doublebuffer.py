#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import time

def blen(line):
    return len(re.sub('\033.*?m', '', line))

def do(cmd, seconds):
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
        print('\033[H', end='')
        print('\n'.join(lines))
        for i in range(linecount, termsize.lines - 1):
            print(' ' * termsize.columns)
        time.sleep(seconds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--interval', metavar='seconds', action='store', type=int, default=5, help='seconds to wait between updates')
    parser.add_argument('command')
    args = parser.parse_args()
    do(args.command, seconds=args.interval)

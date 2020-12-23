#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import time
import datetime

HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'
CLEAR_TERM = '\033[H'
RESET = '\033[0m'

def tokenize(line):
    chars = list(line)
    tokens = []
    while chars:
        c = chars.pop(0)
        if c == '\033':
            token = c
            while chars and c != 'm':
                c = chars.pop(0)
                token += c
            tokens.append(token)
        else:
            tokens.append(c)
    return tokens

def linesplit(oldtokenlines, columns):
    # TODO: decode tab spacing
    counter = 0
    tokenlines = []
    for tokens in oldtokenlines:
        counter = 0
        tokenlines.append([])
        for token in tokens:
            if token.startswith('\033'):
                tokenlines[-1].append(token)
            else:
                if counter == columns:
                    counter = 0
                    tokenlines.append([])
                counter += 1
                tokenlines[-1].append(token)
        tokenlines[-1].append(' ' * (columns - counter))
    tokenlines = [''.join(tokens) for tokens in tokenlines]
    return tokenlines

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
        capture = subprocess.run(cmd, shell=True, capture_output=True)
        lines = capture.stdout.decode().replace('\r\n', '\n').split('\n')
        lines += capture.stderr.decode().replace('\r\n', '\n').split('\n')
        while lines and (not lines[-1] or lines[-1].isspace()):
            lines.pop()

        termsize = os.get_terminal_size()
        tokenlines = [tokenize(line) for line in lines]
        lines = linesplit(tokenlines, termsize.columns)

        more = len(lines) > termsize.lines
        if more:
            lastline = tokenize(lines[termsize.lines - 1])
            lines = lines[:termsize.lines - 1]
            moretext = format('...XmoreX...', f'^{termsize.columns}')[:termsize.columns]
            start = 0
            while moretext[start].isspace():
                start += 1
            end = len(moretext) - 1
            while moretext[end-1].isspace():
                end -= 1
            moretext = moretext.replace('X', ' ')
            i = 0
            text = ''
            lastcode = RESET
            while i < start:
                token = lastline.pop(0)
                text += token
                if token.startswith('\033'):
                    lastcode = token
                else:
                    i += 1
            # dark blue
            text += '\033[38;5;24m'
            while i < end:
                text += moretext[i]
                i += 1
                token = lastline.pop(0)
                while token.startswith('\033'):
                    lastcode = token
                    token = lastline.pop(0)
            text += lastcode
            text += ''.join(lastline)
            lines.append(text)

        print(CLEAR_TERM, end='')
        print('\n'.join(lines), end=RESET)
        for i in range(len(lines), termsize.lines):
            print('\n' + ' ' * termsize.columns, end=RESET)
        if args.on_minute:
            now = datetime.datetime.now()
            elapsed = now.minute % args.on_minute
            elapsed *= 60
            elapsed += now.second + now.microsecond / (10**6)
            remainder = args.on_minute * 60 - elapsed
            time.sleep(remainder)
        else:
            time.sleep(args.interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--interval', metavar='seconds', action='store', type=int, default=5, help='seconds to wait between updates')
    parser.add_argument('-N', '--on-minute', metavar='minute(s)', action='store', type=int, help='update every N minutes on the minute')
    parser.add_argument('-t', '--pseudo-terminal', action='store_true', help='run command using `script` to fake a TTY')
    parser.add_argument('-i', '--interactive-shell', action='store_true', help='run command using `bash -i`')
    parser.add_argument('command')
    args = parser.parse_args()
    if args.on_minute:
        if 60 % args.on_minute:
            print('no')
            exit(1)
    print(HIDE_CURSOR, end='')
    try:
        do(args)
    finally:
        print(SHOW_CURSOR, end='', flush=True)

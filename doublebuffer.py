#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import datetime
import signal

import asyncio

HIDE_CURSOR = '\033[?25l'
SWITCH_TO_ALT = '\033[?1049h'
CLEAR_TERM = '\033[H'
SWITCH_TO_NORM = '\033[?1049l'
SHOW_CURSOR = '\033[?25h'
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

def linesplit(oldtokenlines, columns, chop=False):
    counter = 0
    tokenlines = []
    for tokens in oldtokenlines:
        counter = 0
        tokenlines.append([])
        rtokens = list(reversed(tokens))
        while rtokens:
            token = rtokens.pop()
            if token.startswith('\033'):
                tokenlines[-1].append(token)
            elif token == '\t':
                padding = 8 - (counter % 8)
                rtokens.extend(' ' * padding)
            else:
                if counter == columns:
                    counter = 0
                    tokenlines.append([])
                counter += 1
                tokenlines[-1].append(token)
                if counter == columns and chop:
                    break
        tokenlines[-1].append(' ' * (columns - counter))
    tokenlines = [''.join(tokens) for tokens in tokenlines]
    return tokenlines

def quote(s):
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return s

debug = False
template_cmd = None

repaint_queue = None
run_cmd_queue = None

lines_lock = None
out_lines = []

async def run_cmd():
    global debug
    global template_cmd
    global lines_lock
    global out_lines
    global run_cmd_queue
    global repaint_queue

    while await run_cmd_queue.get():
        termsize = os.get_terminal_size()
        cmd = template_cmd
        cmd = cmd.replace('%W', str(termsize.columns))
        cmd = cmd.replace('%H', str(termsize.lines))
        cmd = re.sub(r'%{H\+(\d+)}', lambda m: str(termsize.lines + int(m[1])), cmd)

        capture = await asyncio.create_subprocess_shell(cmd,
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await capture.communicate()
        lines = stdout.decode().replace('\r\n', '\n').split('\n')
        lines += stderr.decode().replace('\r\n', '\n').split('\n')
        while lines and (not lines[-1] or lines[-1].isspace()):
            lines.pop()

        if args.debug:
            lines = list(map(repr, lines))

        tokenlines = [tokenize(line) for line in lines]

        async with lines_lock:
            out_lines = tokenlines

        await repaint_queue.put(True)

async def repaint(args):
    global lines_lock
    global out_lines
    global repaint_queue

    while await repaint_queue.get():
        async with lines_lock:
            tokenlines = out_lines[:]

        termsize = os.get_terminal_size()
        lines = linesplit(tokenlines, termsize.columns, args.chop_long_lines)

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

async def timer(args):
    global run_cmd_queue
    while True:
        await run_cmd_queue.put(True)
        if args.on_minute:
            now = datetime.datetime.now()
            elapsed = now.minute % args.on_minute
            elapsed *= 60
            elapsed += now.second + now.microsecond / (10**6)
            remainder = args.on_minute * 60 - elapsed
            await asyncio.sleep(remainder)
        else:
            await asyncio.sleep(args.interval)

async def slow_repaint():
    await repaint_queue.put(True)
    await run_cmd_queue.put(True)

# callback
def sigwinchange_handler(*args):
    global repaint_queue
    asyncio.create_task(slow_repaint())

async def async_do(args):
    global debug
    global lines_lock
    global run_cmd_queue
    global repaint_queue

    debug = args.debug

    lines_lock = asyncio.Lock()
    run_cmd_queue = asyncio.Queue()
    repaint_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGWINCH, sigwinchange_handler)
    await asyncio.gather(
        timer(args),
        run_cmd(),
        repaint(args),
    )

def do(args):
    global template_cmd

    cmd = ' '.join(args.command)
    if args.interactive_shell:
        cmd = f'bash -ic "{quote(cmd)}"'
    if args.pseudo_terminal:
        cmd = f'script -qec "{quote(cmd)}" /dev/null'
    template_cmd = cmd

    asyncio.run(async_do(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--interval', metavar='seconds', action='store', type=int, default=5, help='seconds to wait between updates')
    parser.add_argument('-N', '--on-minute', metavar='minute(s)', action='store', type=int, help='update every N minutes on the minute')
    parser.add_argument('-t', '--pseudo-terminal', action='store_true', help='run command using `script` to fake a TTY')
    parser.add_argument('-i', '--interactive-shell', action='store_true', help='run command using `bash -i`')
    parser.add_argument('-S', '--chop-long-lines', action='store_true', help='truncate instead of wrapping')
    parser.add_argument('--debug', action='store_true', help='print repr instead of lines')
    parser.add_argument('command', nargs='*')

    args = parser.parse_args()

    if args.on_minute:
        if 60 % args.on_minute:
            print('no')
            exit(1)
    print(HIDE_CURSOR, end='', flush=True)
    print(SWITCH_TO_ALT, end='', flush=True)
    try:
        do(args)
    finally:
        print(SWITCH_TO_NORM, end='', flush=True)
        print(SHOW_CURSOR, end='', flush=True)

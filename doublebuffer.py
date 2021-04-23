#!/usr/bin/env python3

import argparse
import asyncio
import concurrent.futures
import datetime
import re
import signal
import subprocess
import threading
from functools import partial

import blessed
from wcwidth import wcwidth

import tailf


def chop(line, width=None):
    if width is None:
        width = term.width
    nline = list(blessed.sequences.iter_parse(term, line))
    length = sum(max(wcwidth(text), 0) for text, cap in nline if cap is None)
    endcaps = []
    while length > width and nline:
        text, cap = nline.pop()
        if cap is None:
            length -= max(wcwidth(text), 0)
        else:
            endcaps.insert(0, text)
    return ''.join(text for text, _ in nline) + ''.join(endcaps)


debug = False
chop_long_lines = False
template_cmd = None

term = blessed.Terminal()
out_lines = []
out_lines_lock = threading.Lock()


def templatize(s):
    s = s.replace('%W', str(term.width))
    s = s.replace('%H', str(term.height))
    s = re.sub(r'%{H\+(\d+)}', lambda m: str(term.height + int(m[1])), s)
    return s


executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_cmd(loop):
    global debug
    global template_cmd
    global out_lines
    global out_lines_lock

    cmd = tuple(templatize(arg) for arg in template_cmd)

    capture = subprocess.run(cmd, capture_output=True)
    stdout, stderr = capture.stdout, capture.stderr
    lines = stdout.decode(errors='ignore').replace('\r\n', '\n').split('\n')
    lines += stderr.decode().replace('\r\n', '\n').split('\n')
    while lines and (not lines[-1] or lines[-1].isspace()):
        lines.pop()

    if debug:
        lines = list(map(repr, lines))

    with out_lines_lock:
        out_lines = lines
    asyncio.run_coroutine_threadsafe(repaint(), loop)


MORE = '... more ...'


async def repaint():
    global out_lines
    global out_lines_lock

    with out_lines_lock:
        lines = out_lines[:]

    lines = (
        list(map(partial(chop, width=term.width), lines)) if chop_long_lines else lines
    )

    more = len(lines) > term.height
    if more:
        lines = lines[: term.height]
        moretext = format(MORE, f'^{term.width}')
        start = moretext.find(MORE)
        moretext = moretext[: term.width].strip()
        moretext = term.move_yx(term.height - 1, start) + term.dodgerblue4 + moretext
        lines[-1] += moretext

    print(term.home, term.normal, term.clear, sep='', end='')
    for i, line in enumerate(lines):
        print(term.move_yx(i, 0), line, sep='', end='')
    print(end='', flush=True)


async def timer(args):
    while True:
        executor.submit(run_cmd, asyncio.get_running_loop())
        if args.on_minute:
            now = datetime.datetime.now()
            elapsed = now.minute % args.on_minute
            elapsed *= 60
            elapsed += now.second + now.microsecond / (10 ** 6)
            remainder = args.on_minute * 60 - elapsed
            await asyncio.sleep(remainder)
        else:
            await asyncio.sleep(args.interval)


# run in separate thread
def watcher(args, loop):
    if args.watch:
        for _ in tailf.watch_file(args.watch):
            executor.submit(run_cmd, loop)


async def slow_repaint():
    asyncio.create_task(repaint())
    executor.submit(run_cmd, asyncio.get_running_loop())


# callback
def sigwinchange_handler(*args):
    asyncio.create_task(slow_repaint())


async def async_do(args):
    global debug

    debug = args.debug

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGWINCH, sigwinchange_handler)
    await asyncio.gather(asyncio.to_thread(watcher, args, loop), timer(args))


def do(args):
    global template_cmd

    cmd = tuple(args.command)
    if args.interactive_shell:
        cmd = ('bash', '-ic', ' '.join(cmd))
    if args.pseudo_terminal:
        cmd = ('script', '-qec', ' '.join(cmd), '/dev/null')
    template_cmd = cmd

    asyncio.run(async_do(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--interval',
        metavar='seconds',
        action='store',
        type=int,
        default=5,
        help='seconds to wait between updates',
    )
    parser.add_argument(
        '-N',
        '--on-minute',
        metavar='minute(s)',
        action='store',
        type=int,
        help='update every N minutes on the minute',
    )
    parser.add_argument(
        '-w',
        '--watch',
        metavar='FILE',
        action='store',
        help='update when watched file changes',
    )
    parser.add_argument(
        '-t',
        '--pseudo-terminal',
        action='store_true',
        help='run command using `script` to fake a TTY',
    )
    parser.add_argument(
        '-i',
        '--interactive-shell',
        action='store_true',
        help='run command using `bash -i`',
    )
    parser.add_argument(
        '-S',
        '--chop-long-lines',
        action='store_true',
        help='truncate instead of wrapping',
    )
    parser.add_argument(
        '--debug', action='store_true', help='print repr instead of lines'
    )
    parser.add_argument('command', nargs='*')

    args = parser.parse_args()

    # global chop_long_lines
    chop_long_lines = args.chop_long_lines

    if args.on_minute:
        if 60 % args.on_minute:
            print('no')
            exit(1)

    with term.fullscreen(), term.hidden_cursor(), term.cbreak():
        do(args)

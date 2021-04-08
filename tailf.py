import contextlib
import os.path

from inotify_simple import INotify, flags


def watch_file(path):
    [watchpath, watchname] = os.path.split(path)
    with contextlib.closing(INotify()) as inotify:
        inotify.add_watch(watchpath or '.', flags.CLOSE_WRITE | flags.MOVED_TO)
        while True:
            for event in inotify.read():
                if event.name == watchname:
                    yield event.name


def contents_on_change(path, mode='r', *, callback):
    for _ in watch_file(path):
        with open(path, mode) as f:
            yield callback(f)


def sub_blocking(path, mode='r'):
    for contents in contents_on_change(path, mode, callback=lambda f: f.read()):
        yield (contents.splitlines() if mode == 'r' else contents)

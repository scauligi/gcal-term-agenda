#!/usr/bin/env python3

import argparse
import calendar
import os
import pickle
import re
import sys
from collections import Counter
from collections import defaultdict as ddict
from collections import namedtuple
from datetime import date, datetime, time
from datetime import timedelta as t

import blessed
from dateutil.tz import tzlocal

import gcal
from gcal import Event, as_date, as_datetime, base_datetime

TerminalSize = namedtuple(
    'TerminalSize', ['columns', 'lines', 'columns_auto', 'lines_auto']
)

# https://stackoverflow.com/a/43950235
# Monkey patch to force IPv4
def ipv4_monkey_patch():
    import socket

    orig_gai = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return orig_gai(
            host, port, family=socket.AF_INET, type=type, proto=proto, flags=flags
        )

    socket.getaddrinfo = getaddrinfo


# https://stackoverflow.com/a/56842689
class reversor:
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        return other.obj < self.obj


term = blessed.Terminal()
term.number_of_colors = 256 ** 3


def fg(short):
    if isinstance(short, tuple):
        return term.color_rgb(*short)
    return f'\033[38;5;{short}m'


LGRAY = fg(250)
MLGRAY = fg(243)
MMLGRAY = fg(240)
MGRAY = fg(237)
DGRAY = fg(235)
NOWLINE_COLOR = fg(66)
RESET = '\033[0m'
WBOLD = '\033[0;1m'


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


def blen(line):
    return len(re.sub('\033.*?m', '', line))


def shorten(text, inner_width):
    if len(text) > inner_width:
        text = text[: inner_width - 1] + '⋯'
    return f'{text:<{inner_width}}'


def bshorten(text, max_width):
    text = text.rstrip()
    textlen = blen(text)
    if textlen > max_width:
        lastchar = None
        tokens = tokenize(text)
        while tokens and textlen > max_width - 1:
            if not (lastchar := tokens.pop()).startswith('\033'):
                textlen -= 1
        if lastchar == '─':
            # hacky!
            tokens.append('┄')
        else:
            tokens.append('⋯')
        return ''.join(tokens) + RESET
    return text


def _strippable(token):
    return (token.isspace() and token != '\xa0') or token.startswith('\033')


def brstrip(text):
    tokens = tokenize(text)
    while tokens and _strippable(tokens[-1]):
        tokens.pop()
    return ''.join(tokens) + RESET


# overlay `text` onto `row` at index `offset`
def place(text, offset, row):
    if '\0' in text:
        texts = text.split('\0')
        lastcode = ''
        for text in texts:
            text = lastcode + text
            row = place(lastcode + text, offset, row)
            for tok in reversed(tokenize(text)):
                if tok.startswith('\033'):
                    lastcode = tok
                    break
            offset += blen(text) + 1
        return row
    rowlen = blen(row)
    if rowlen <= offset:
        row += ' ' * (offset - rowlen)
        return row + text
    elif offset + blen(text) < rowlen:
        tokens = tokenize(row)
        aftertokens = []
        lastcode = None
        while rowlen > offset:
            tok = tokens.pop()
            aftertokens.append(tok)
            if not tok.startswith('\033'):
                rowlen -= 1
        if lastcode is None:
            for tok in reversed(tokens):
                if tok.startswith('\033'):
                    lastcode = tok
                    break
            else:
                lastcode = RESET
        rowlen = blen(text)
        while rowlen > 0:
            tok = aftertokens.pop()
            if not tok.startswith('\033'):
                rowlen -= 1
            else:
                lastcode = tok
        return (
            ''.join(tokens) + RESET + text + lastcode + ''.join(reversed(aftertokens))
        )
    else:
        tokens = tokenize(row)
        while rowlen > offset:
            if not tokens.pop().startswith('\033'):
                rowlen -= 1
        return ''.join(tokens) + RESET + text


def dtime(dt=None, shrink=False):
    if not dt:
        return re.sub(r'.', ' ', date.today().strftime('%a %m/%d'))
    s = dt.strftime('%a %m/%d')
    s = re.sub(r'0(\d)/', r' \1/', s)
    s = re.sub(r'/0(\d)$', r'/\1 ', s)
    if shrink:
        s = s.rstrip()
        s = re.sub(r' +', r' ', s)
    return s


def ftime(dt=None, now=False):
    if not dt:
        return '      '
    s = dt.strftime('%I:%M%P')
    s = re.sub(r'm$', '', s)
    s = re.sub(r'^0', ' ', s)
    if not now:
        s = re.sub(r':00([ap])', r'\1   ', s)
        if not dt.minute:
            if dt.hour == 0:
                s = 'mdnght'
            elif dt.hour == 12:
                s = ' noon '
    return s


def tdtime(td, prefer_hours=False):
    days = td.days
    minutes = td.seconds // 60
    hours, minutes = divmod(minutes, 60)
    if prefer_hours and (hours or minutes):
        hours += 24 * days
        days = 0
    s = ''
    if days:
        s += str(days) + ' day'
        s += 's' if days != 1 else ''
    if hours:
        s += ', ' if s else ''
        s += str(hours) + ' hour'
        s += 's' if hours != 1 else ''
    if minutes:
        s += ', ' if s else ''
        s += str(minutes) + ' minute'
        s += 's' if minutes != 1 else ''
    return s


def filter_calendars(obj, calendars):
    callist = []
    allCals = gcal.get_visible_cals(obj['calendars'])
    if not calendars:
        calendars = 'all'

    def _getCal(calendar):
        if isinstance(calendar, list):
            list(map(_getCal, calendar))
        elif '@' not in calendar:
            lookup = allCals.get(calendar)
            if lookup is None:
                print('Error: unknown calendar:', calendar, file=sys.stderr)
            _getCal(lookup)
        else:
            callist.append(calendar)

    _getCal(calendars)
    return callist


def string_outofdate(obj, now=None):
    if not now:
        now = datetime.now(tzlocal())
    if obj['timestamp'] + t(minutes=24 * 60) < now:
        return (
            fg(3)
            + 'Warning: timestamp is out of date by '
            + tdtime(now - obj['timestamp'])
            + RESET
        )
    return None


def make_evt2short(obj):
    cal2short = {}
    cal2dark = {}
    for cal in obj['calendars']:
        code = cal['backgroundColor']
        rgb = [int(code[x : x + 2], 16) for x in (1, 3, 5)]
        v = max(rgb)
        new_v = min(max(v - 70, 0), 127)
        scaling = new_v / v
        dark_rgb = [round(x * scaling) for x in rgb]
        cal2short[cal['id']] = tuple(rgb)
        cal2dark[cal['id']] = tuple(dark_rgb)

    def evt2short(evt, dark=False):
        if dark:
            return cal2dark[evt.calendar]
        return cal2short[evt.calendar]

    return evt2short


def get_events(obj, todate, ndays, callist, local_recurring=False):
    events = []
    callist = list(set(callist))

    if ndays >= 0:
        rows = obj['db'].execute(
            f"""
              select id,blob,local_recurring,start from events_recurring
              where
                    date(enddate) >= date(?)
                    and date(startdate) < date(?, "+" || ? || " days")
                 and not cancelled
                 and calendar in ({','.join("?"*len(callist))})
                order by startdate, hastime, datetime(start), datetime(end) desc
        """,
            (todate, todate, ndays, *callist),
        )
    else:
        rows = obj['db'].execute(
            f"""
              select root_id,blob,local_recurring,min(start) from events_recurring
              where
                    date(enddate) >= date(?)
                 and not cancelled
                 and calendar in ({','.join("?"*len(callist))})
                group by root_id
                order by startdate, hastime, datetime(start), datetime(end) desc
        """,
            (todate, *callist),
        )

    for _, blob, recurs_locally, _ in rows:
        ev = Event.unpkg(pickle.loads(blob))
        if local_recurring:
            ev.recurring = bool(recurs_locally)
        events.append(ev)
    return events


class Agenda:
    def __init__(self, calendars, objs, dark_recurring=False, interval=None):
        self.now = datetime.now(tzlocal())
        self.obj, evt2short = objs

        def _evt2short(evt):
            if as_date(evt.start) > as_date(self.now):
                dark = dark_recurring and evt.recurring
            else:
                dark = as_datetime(evt.end) <= self.now
            return evt2short(evt, dark=dark)

        self.evt2short = _evt2short
        self.callist = filter_calendars(self.obj, calendars)
        self.interval = t(minutes=(interval or 15))
        self.seen_events = set()

    # assumes times with granularity at minutes
    def quantize(self, thetime, endtime=False):
        if endtime:
            return self.quantize(thetime - t(minutes=1))
        minutes = self.interval.seconds // 60
        theminutes = (thetime.hour * 60 + thetime.minute) // minutes * minutes
        return datetime(
            thetime.year,
            thetime.month,
            thetime.day,
            theminutes // 60,
            theminutes % 60,
            tzinfo=thetime.tzinfo,
        )

    def agenda_table(self, todate, ndays=None, min_start=None, print_warning=True):
        self.todate = todate
        self.has_later = False

        if min_start is None:
            min_start = time(tzinfo=tzlocal())
        else:
            min_start = self.quantize(datetime.combine(todate, min_start)).time()

        # table[0] is the time column
        # table[n] is the event column for day n
        # each column is a map keyed by tick
        # special key None is for full-day events
        # also table[0][None] is reserved for status messages (like the outofdate string)
        #
        # table[0][None]: list[str]  (status messages that should print before the table)
        # table[0][tick]: bool       (whether there is an event starting at this time anywhere in the table
        # table[n][None]: list[evt]  (list of full-day events)
        # table[n][tick]: list[evt]  (list of events starting at that tick)
        actual_ndays = ndays or 1
        table = [ddict(lambda: None)]
        table[0][None] = []
        for _ in range(actual_ndays):
            table.append(ddict(list))

        if print_warning:
            if outofdate := string_outofdate(self.obj, self.now):
                table[0][None].append(outofdate)

        events = get_events(
            self.obj, todate, actual_ndays, self.callist, local_recurring=True
        )

        longest_fullday_summary = max(
            (len(evt.summary) for evt in events if not isinstance(evt.start, datetime)),
            default=0,
        )
        longest_timeblock_summary = max(
            (len(evt.summary) for evt in events if isinstance(evt.start, datetime)),
            default=0,
        )
        self.longest_summary = (longest_fullday_summary, longest_timeblock_summary)

        for evt in events:
            # get column (1-indexed)
            # accounts for events that start before the first day
            startindex = (as_date(evt.start) - todate).days
            index = max(startindex, 0) + 1
            if not isinstance(evt.start, datetime):
                # full-day event
                if evt.id not in self.seen_events:
                    table[index][None].append(evt)
                    self.seen_events.add(evt.id)
                continue
            # timeblock event
            if startindex >= 0:
                tickt = self.quantize(evt.start).time()
            else:
                tickt = min_start
            tickt = max(min_start, tickt)
            if evt.end.time() > min_start or evt.end.date() > evt.start.date():
                table[0][tickt] = tickt
                table[index][tickt].append(evt)

        return table

    # convenience for looping over changing column(s)
    def _intervals(self, *cols, start_min=False):
        if not any(cols):
            return
        minutes = 0
        tickt = time()
        if start_min:
            tickt = min(min(col.keys(), default=time()) for col in cols)
            if isinstance(start_min, time):
                tickt = min(start_min, tickt)
            minutes = tickt.hour * 60 + tickt.minute
        while tickt <= max(max(col.keys(), default=time()) for col in cols):
            yield tickt
            minutes += self.interval.seconds // 60
            if minutes >= 24 * 60:
                break
            tickt = time(*divmod(minutes, 60))

    # convenience for looping until an end time
    def _until(self, starttick, endtick):
        tick = starttick
        while tick < endtick:
            yield tick
            tick += self.interval

    def _evtcol(self, *evtcols, forced, locations, nowtick=None):
        # whether to show anchors/endings and location or not
        def _expand(evt):
            if forced:
                return True
            expand = nowtick < self.quantize(evt.end)
            self.has_later = expand or self.has_later
            return expand

        # iterate over each interval slot
        contents = ddict(list)  # tick -> (col_index, initial, summary)
        for tickt in self._intervals(*evtcols):
            for col_index, evtcol in enumerate(evtcols):
                tick = base_datetime(self.todate + t(days=col_index), tickt)
                tock = tick + self.interval

                for evt in evtcol[tickt]:
                    expand = _expand(evt)
                    summary = evt.summary
                    endtext = ''

                    if locations:
                        locstrs = []
                        if evt.location:
                            locstrs.append(' '.join(evt.location.split()))
                        if evt.link and evt.link not in locstrs:
                            locstrs.append(' '.join(evt.link.split()))

                    # display true start time if necessary
                    if evt.start != tick:
                        endtext += ' ({})'.format(ftime(evt.start).strip())

                    # show ending markers on short events
                    if expand:
                        if evt.end < tock:
                            if evt.end == evt.start:
                                endtext += ' <-'
                            else:
                                if endtext:
                                    endtext = endtext[:-1] + ' -> {})'
                                else:
                                    endtext += ' (-> {})'
                                endtext = endtext.format(ftime(evt.end).strip())
                        if locations:
                            remaining_ticks = (
                                self.quantize(evt.end) - tock
                            ).seconds // self.interval.seconds
                            locstrs_to_join = max(len(locstrs) - remaining_ticks, 0)
                            locstr = ' '.join(locstrs[:locstrs_to_join])
                            if locstr:
                                endtext = ' ' + locstr + endtext
                            locstrs = locstrs[locstrs_to_join:]

                    summary = (
                        fg(self.evt2short(evt)) + summary + endtext + RESET + '   '
                    )

                    # place into leftmost region that is large enough
                    prev_end = 0
                    for (icol_index, initial, text) in sorted(contents[tickt]):
                        if icol_index != col_index:
                            continue
                        if initial - prev_end >= blen(summary):
                            break
                        prev_end = initial + blen(text)
                    contents[tickt].append((col_index, prev_end, summary))
                    initial = prev_end

                    # drop anchor on long events
                    if expand:
                        for endtick in self._until(tock, evt.end):
                            col_offset = (endtick.date() - tick.date()).days
                            endtock = endtick + self.interval
                            if evt.end == endtock:
                                # \xa0 is non-breaking space
                                # which is handled specially in brstrip()
                                text = '\033[4m\xa0│\xa0\033[24m\0'
                            elif evt.end < endtock:
                                text = '┴┴┴ ({}) '.format(ftime(evt.end).strip())
                            else:
                                text = '\0│\0\0'
                            if locations and locstrs:
                                text += locstrs.pop(0) + '  '
                            text = fg(self.evt2short(evt)) + text + RESET
                            contents[endtick.time()].append(
                                (col_index + col_offset, initial, text)
                            )
                            endtick += self.interval
        for tickt in contents:
            contents[tickt].sort()
        return contents

    def dayview(self, table, forced=False):
        newtable = []

        nowtick = self.quantize(self.now)
        timefield = ftime().replace(' ', '-')
        did_first = False

        # expect table with only [0] and [1] columns
        timecol, evtcol = table

        # whether to do "now" arrow
        is_todate = self.now.date() == self.todate

        # if there are no actual events
        if not any(evtcol.items()):
            newtable.append(
                f'{LGRAY}{dtime(self.todate)} {timefield}  no events{RESET}'
            )
            # place the "now" arrow
            if is_todate:
                newtable.append(
                    f'{dtime()} {ftime()}    <-- ' + ftime(self.now, now=True)
                )

            return newtable

        # print each full-day
        fulldays = evtcol.pop(None, [])
        for evt in fulldays:
            span = ''
            ndays = (evt.end - evt.start).days
            if ndays != 1:
                startspan = ''
                endspan = ''
                lastdate = evt.end - t(days=1)
                if evt.start != self.todate:
                    startspan = dtime(evt.start, shrink=True) + ' '
                if lastdate != self.todate:
                    endspan = ' ' + dtime(lastdate, shrink=True)
                span = LGRAY + ' ({}->{})'.format(startspan, endspan)
            summary = fg(self.evt2short(evt)) + evt.summary
            newtable.append([timefield, summary + span + RESET])

        contents = self._evtcol(evtcol, forced=forced, nowtick=nowtick, locations=True)

        lasttick = max(contents.keys(), default=time())
        if is_todate:
            # make sure there's a ddict key entry for nowtick
            # so that self._intervals iterates at least that far
            contents[nowtick.time()]

        # assemble newtable from contents
        for tickt in self._intervals(contents):
            tick = base_datetime(self.todate, tickt)

            # skip blank slots until the first event
            if (
                not did_first
                and not timecol[tickt]
                and not (is_todate and tick == nowtick)
            ):
                continue
            # skip blank slots after the last event
            if tickt > lasttick and not (is_todate and tick == nowtick):
                continue
            # skip blank slots today before "now" (if not forced)
            if not forced and is_todate and tick < nowtick and not contents[tickt]:
                continue

            # print tick time for event starts
            timefield = ftime(tick if timecol[tickt] else None)

            # print each event at the right output column
            content = ''
            prev_end = 0
            for (col_index, initial, text) in contents[tickt]:
                if col_index == 0:
                    content += ' ' * (initial - prev_end)
                    content += text
                    prev_end = initial + blen(text)
                    did_first = True

            # place the "now" arrow
            if is_todate and tick == nowtick:
                content += '  <-- ' + ftime(self.now, now=True)
                did_first = True

            newtable.append([timefield, content])

        # compile newtable
        newtable = [[dtime(), *line] for line in newtable]
        newtable[0][0] = dtime(self.todate)
        newtable = [
            f'{LGRAY}{line[0]} {line[1]}  {line[2]}{RESET}' for line in newtable
        ]

        # collect status messages
        if timecol[None]:
            newtable = timecol[None] + newtable

        return newtable

    @staticmethod
    def print_table(table):
        newline = '(\n(' + re.escape(RESET) + ')?)'
        lines = '\n'.join(brstrip(line) for line in table)
        lines = lines.replace('\0', ' ')
        lines = re.sub(newline + r'{9,}', r'\n' * 9 + RESET, lines)
        lines = re.sub(newline + r'*$', '', lines)
        print(lines)


def listcal(todate, calendars, no_recurring=False, forced=False, objs=None):
    today = base_datetime(todate)
    now = datetime.now(tzlocal())
    obj, evt2short = objs

    table = []
    if outofdate := string_outofdate(obj, now):
        table.append(outofdate)

    callist = filter_calendars(obj, calendars)

    weekday = todate.isoweekday() % 7
    nextsunday = 7 - weekday
    nextmonth = date(today.year + today.month // 12, today.month % 12 + 1, 1)
    nextmonth = base_datetime(nextmonth)
    follmonth = date(
        today.year + (today.month + 1) // 12, (today.month + 1) % 12 + 1, 1
    )
    follmonth = base_datetime(follmonth)

    highwater = [
        (None, '== ONGOING =='),
        (today, '== TODAY =='),
        (today + t(days=1), '== TOMORROW =='),
        (today + t(days=2), '== THIS WEEK =='),
        (today + t(days=nextsunday), '== NEXT WEEK =='),
        (today + t(days=(nextsunday + 7)), '== FOLLOWING WEEK =='),
        (today + t(days=(nextsunday + 14)), '== THIS MONTH =='),
        (nextmonth, '== NEXT MONTH =='),
        (follmonth, '== THE FUTURE =='),
    ]
    if forced:
        highwater[1] = (today, f'== {today.strftime("%b %d").upper()} ==')
        del highwater[2:]

    events = get_events(
        obj, todate, 1 if forced else -1, callist, local_recurring=forced
    )

    seen = Counter()
    for evt in events:
        if no_recurring and evt.recurring:
            continue
        start = as_datetime(evt.start)
        seen[evt.uid] += 1
        if seen[evt.uid] > 1:
            continue
        s = None
        while highwater and (highwater[0][0] is None or start >= highwater[0][0]):
            _, s = highwater.pop(0)
        if s:
            table.append(fg(4) + s + RESET)
        dark = isinstance(evt.end, datetime) and evt.end < now
        gray = fg(8) if dark else LGRAY
        white = fg(8) if dark else ''
        fmt = '  '
        fmt += gray + dtime(evt.start) + RESET
        fmt += ' '
        fmt += gray + ftime(evt.start if isinstance(evt.start, datetime) else None)
        fmt += ' '
        fmt += fg(evt2short(evt, dark=dark)) + evt.summary + RESET
        fmt += ' '
        fmt += (
            white
            + re.sub(r'^1 day$', r'', tdtime(evt.end - evt.start, prefer_hours=True))
            + RESET
        )
        table.append(fmt)
    return table


# fmt: off
CORNERS = """\
┌┬┐▄
├┼┤█
└┴┘▀""".split('\n')
DASH = "─"
PIPE = "│"
THICK = "█"
# fmt: on


def fourweek(
    todate,
    calendars,
    termsize=None,
    objs=None,
    zero_offset=False,
    table_height=4,
    table_cells=None,
    no_recurring=False,
):
    table_width = 7

    table = []

    offset = todate.isoweekday() % table_width
    thick_index = 0
    if zero_offset:
        thick_index = (table_width - offset) % table_width
        offset = 0

    roffset = 0
    if table_cells is not None:
        table_height = (table_cells + offset + (table_width - 1)) // table_width
        roffset = table_width - ((table_width * table_height) - (offset + table_cells))

    linecolor = MMLGRAY if not no_recurring else MGRAY

    def do_row(i):
        fill = DASH
        corner_index = 0 if i == 0 else 2 if i == table_height else 1

        flips = [[0, corner_index], [table_width + 1, -1]]

        if table_cells and offset:
            if i == 0 or i == 1 == table_height:
                flips[0][0] = offset
            elif i == 1:
                flips[0][1] = 0
                flips.append([offset, 1])
        if roffset:
            if i == table_height or i == table_height - 1 == 0:
                flips.append([roffset + 1, -1])
            elif i == table_height - 1:
                flips.append([roffset + 1, 2])

        flips.sort()

        start, index = flips.pop(0)
        line = " " * start * (inner_width + 1)
        left, mid, right, thick = CORNERS[index]
        line += left
        for end, index in flips:
            end -= 1
            for j in range(start, end):
                line += fill * inner_width
                line += thick if thick_index and thick_index == j else mid
            start = end
            if index < 0:
                break
            left, mid, right, thick = CORNERS[index]
        line = line[:-1] + right
        return linecolor + line + RESET

    obj, _evt2short = objs
    now = datetime.now(tzlocal())
    nowdate = now.date()

    def evt2short(evt):
        if table_cells and (
            as_date(evt.end) < todate
            or as_date(evt.start) >= todate + t(days=table_cells)
        ):
            dark = True
        elif as_date(evt.start) > as_date(now):
            dark = evt.recurring
        else:
            dark = as_datetime(evt.end) <= now
        return _evt2short(evt, dark=dark)

    def choice(evt):
        if as_date(evt.start) == nowdate:
            return isinstance(evt.end, datetime) and evt.start >= now
        else:
            return evt.recurring

    callist = filter_calendars(obj, calendars)

    OPEN = object()
    CLOSED = object()
    weekcells = [
        [ddict(lambda: OPEN) for _ in range(table_width)] for _ in range(table_height)
    ]
    cells = [(list(), list()) for _ in range(table_width * table_height)]

    calstart = todate - t(days=offset)
    events = get_events(
        obj,
        todate - t(days=offset),
        table_width * table_height,
        callist,
        local_recurring=True,
    )

    running_width = 0
    ftime_len = len(ftime()) + 1
    for evt in events:
        # XXX do DRY with following loop
        if no_recurring and evt.recurring:
            continue
        cellnum = (as_date(evt.start) - calstart).days
        cellend = (as_date(evt.end) - calstart).days
        if (0 <= cellnum < len(cells)) or (0 < cellend <= len(cells)):
            length = len(evt.summary)
            if isinstance(evt.start, datetime):
                length += ftime_len
            running_width = max(length, running_width)
    if termsize.columns is None:
        inner_width = running_width
    else:
        inner_width = (termsize.columns - (table_width + 1)) // table_width
        inner_width = max(inner_width, 0)
        if termsize.columns_auto:
            inner_width = min(inner_width, running_width)

    for evt in events:
        if no_recurring and evt.recurring:
            continue
        cellnum = (as_date(evt.start) - calstart).days
        cellend = (as_date(evt.end) - calstart).days
        if (0 <= cellnum < len(cells)) or (0 < cellend <= len(cells)):
            cellnum = max(0, cellnum)
            if isinstance(evt.start, datetime):
                text = ftime(evt.start) + ' ' + evt.summary
                text = shorten(text, inner_width)
                text = fg(evt2short(evt)) + text + RESET
                cells[cellnum][choice(evt)].append(text)
            else:
                # full-day event
                # XXX code copied from weekview, need to DRY
                start_week = cellnum // table_width
                end_week = ((evt.end - calstart).days - 1) // table_width
                end_week = min(end_week, table_height - 1)
                for week in range(start_week, end_week + 1):
                    text = ' ' + evt.summary
                    week_start = calstart + t(days=(week * table_width))
                    week_end = week_start + t(days=table_width)

                    incellnum = (max(evt.start, week_start) - week_start).days
                    excellnum = (min(evt.end, week_end) - week_start).days
                    ndays = excellnum - incellnum

                    daycells = weekcells[week]
                    subcells = daycells[incellnum:excellnum]
                    topslot = max(map(len, subcells))
                    for j in range(topslot):
                        if all(
                            j >= len(daycell) or daycell[j] is OPEN
                            for daycell in subcells
                        ):
                            # found an empty slot range
                            break
                    else:
                        j = topslot

                    if evt.start < week_start:
                        text = '┄' + text

                    if (evt.end - evt.start).days > 1:
                        outlen = ndays * (inner_width + 1) - 1
                        text += (
                            ' '
                            + DASH * (outlen - len(text) - 2)
                            + ('┄' if evt.end > week_end else '>')
                        )
                        text = shorten(text, outlen)
                    else:
                        text = shorten(text, inner_width)

                    daycell = subcells[0]
                    assert daycell[j] is OPEN
                    daycell[j] = fg(evt2short(evt)) + text + RESET

                    for daycell in subcells[1:]:
                        assert daycell[j] is OPEN
                        daycell[j] = CLOSED

    filled_cells = []

    for i in range(table_height):
        daycells = weekcells[i]
        for j in range(table_width):
            daycell = daycells[j]
            cell, cell_recurring = cells[i * table_width + j]
            filled_cell = []
            for k in range(max(daycell.keys(), default=-1) + 1):
                text = daycell[k]
                if text is OPEN:
                    text = ' ' * inner_width
                elif text is CLOSED:
                    text = None
                filled_cell.append(text)
            filled_cell.extend(cell)
            if filled_cell:
                filled_cell.append(' ' * inner_width)
            filled_cell.extend(cell_recurring)
            filled_cells.append(filled_cell)

    max_inner_height = max(len(cell) for cell in filled_cells) + 2
    if termsize.lines is None:
        inner_height = max_inner_height
    else:
        inner_height = (termsize.lines - (table_height + 1)) // table_height
        inner_height = max(inner_height, 0)
        if termsize.lines_auto:
            inner_height = min(inner_height, max_inner_height)

    # set up table borders
    line = do_row(0)
    table.append(line)
    for i in range(1, table_height + 1):
        for _ in range(inner_height):
            table.append([])
        line = do_row(i)
        table.append(line)

    # overwrite table with content of cells
    for i in range(table_height):
        for j in range(table_width):
            cell = filled_cells[i * table_width + j]
            has_events = cells[i * table_width + j][0] or any(
                slot is not CLOSED for slot in weekcells[i][j].values()
            )
            for k in range(inner_height):
                lineIndex = i * (inner_height + 1) + k + 1
                text = ' ' * inner_width
                if k == 0:
                    celldate = todate + t(days=(i * table_width + j - offset))
                    datetext = dtime(celldate)
                    dcolor = (
                        MGRAY
                        if table_cells
                        and (
                            i == 0
                            and j < offset
                            or (i == table_height - 1 and j >= roffset)
                        )
                        else LGRAY
                        if has_events
                        else MLGRAY
                        if weekcells[i][j]
                        else MGRAY
                    )
                    if celldate == now.date():
                        datetext = f'> {datetext} <'
                        dcolor = WBOLD
                    text = (
                        dcolor
                        + shorten(f'{datetext:^{inner_width}}', inner_width)
                        + RESET
                    )
                elif k == 1:
                    text = linecolor + DASH * inner_width + RESET
                else:
                    k -= 2
                    if k + 2 == inner_height - 1 and len(cell) > k + 1:
                        text = shorten(
                            format('... more ...', f'^{inner_width}'), inner_width
                        )
                        text = fg(4) + text + RESET
                    elif k < len(cell):
                        text = cell[k]
                table[lineIndex].append(text)

    newtable = []
    for i, line in enumerate(table):
        topline = 2
        top = inner_height
        botline = len(table) - inner_height
        bottom = botline - 2
        if isinstance(line, list):
            text = ""
            for j, segment in enumerate(line):
                left = j < offset
                right = j > roffset

                pipe = PIPE
                if table_cells:
                    if (i <= top and left) or (i > bottom and right):
                        pipe = " "
                    elif thick_index and thick_index == j:
                        pipe = THICK
                    if (i == topline and left) or (i == botline and j > roffset - 1):
                        segment = " " * inner_width

                if segment is not None:
                    text += linecolor + pipe + RESET
                    text += segment

            pipe = " " if table_cells and i > bottom else PIPE
            text += linecolor + pipe + RESET
            newtable.append(text)
        else:
            newtable.append(line)
    if outofdate := string_outofdate(obj, now):
        newtable[0] = place(outofdate, 2, newtable[0])
    return newtable


def weekview(
    todate,
    week_ndays,
    calendars,
    termsize=None,
    objs=None,
    dark_recurring=False,
    zero_offset=False,
    interval=None,
    min_start=None,
    max_start=None,
):
    table_width = week_ndays if week_ndays > 0 else 7
    interval = interval or 30

    offset = 0
    if week_ndays == 0:
        offset = todate.isoweekday() % 7
        if zero_offset:
            offset = (offset - 1) % 7
        week_ndays = 7
    weekstart = todate - t(days=offset)

    agendamaker = Agenda(
        calendars, objs=objs, dark_recurring=dark_recurring, interval=interval
    )
    if min_start is not None:
        min_start = time(*min_start, tzinfo=tzlocal())
    if max_start is not None:
        max_start = time(*max_start, tzinfo=tzlocal())
    timecol, *evtcols = agendamaker.agenda_table(
        weekstart,
        ndays=table_width,
        min_start=min_start,
    )
    if max_start is not None:
        max_start = agendamaker.quantize(datetime.combine(weekstart, max_start)).time()
        timecol[max_start] = max_start

    todate_offset = (agendamaker.now.date() - weekstart).days
    has_todate = 0 <= todate_offset < week_ndays
    nowtick = agendamaker.quantize(agendamaker.now).time()

    timecolsz = len(ftime()) + 1
    longest_summary = max(
        agendamaker.longest_summary[0] + 2, agendamaker.longest_summary[1]
    )
    if termsize.columns is None:
        inner_width = longest_summary
    else:
        inner_width = (termsize.columns - timecolsz - 1) // table_width - 1
        inner_width = max(inner_width, 0)
        if termsize.columns_auto:
            inner_width = min(longest_summary, inner_width)
    outer_width = (inner_width + 1) * table_width + 1 + timecolsz

    # full-day events part 1
    OPEN = object()
    CLOSED = object()
    daycols = [ddict(lambda: OPEN) for evtcol in evtcols]
    for i, evtcol in enumerate(evtcols):
        for evt in evtcol.pop(None, []):
            ndays = (evt.end - weekstart).days - i
            subcols = daycols[i : i + ndays]
            topslot = max(map(len, subcols))
            for j in range(topslot):
                if all(j >= len(daycol) or daycol[j] is OPEN for daycol in subcols):
                    # found an empty slot range
                    break
            else:
                j = topslot

            summary = ' ' + evt.summary
            if evt.start < weekstart:
                summary = '┄' + DASH * timecolsz + summary
            if (evt.end - evt.start).days > 1:
                outlen = ndays * (inner_width + 1) - 1
                summary += ' ' + DASH * (outlen - len(evt.summary) - 3) + '>'

            daycol = subcols[0]
            assert daycol[j] is OPEN
            daycol[j] = fg(agendamaker.evt2short(evt)) + summary + RESET
            if evt.start < weekstart:
                daycol[j] = (-(timecolsz + 1), daycol[j])

            for daycol in subcols[1:]:
                assert daycol[j] is OPEN
                daycol[j] = CLOSED

    final_i = ddict(set)
    for i, evtcol in enumerate(evtcols):
        for tickt in evtcol:
            final_i[tickt].add(i)

    contents = agendamaker._evtcol(*evtcols, forced=True, locations=False)

    newtable = []

    def do_row(tickt, fill, left, mid=None, right=None, _thick=None):
        if mid is None:
            mid = left
        if right is None:
            right = left
        line = left
        for _ in range(table_width):
            line += fill * inner_width + mid
        line = line[:-1] + right
        if isinstance(tickt, str):
            timestr = tickt * timecolsz
        else:
            timestr = '{:>{}}'.format(ftime(tickt).strip() + ' ', timecolsz)
        return LGRAY + timestr + line + RESET

    def assemble_row(tickt, iterable_or_fn, row=None):
        def calc_initial(i, initial):
            return i * (inner_width + 1) + 1 + initial + timecolsz

        if row is None:
            row = do_row(tickt, ' ', PIPE)
            if not isinstance(tickt, str) and tickt:
                max_index = max(final_i[tickt], default=-1)
                fill = DGRAY + DASH * inner_width + RESET
                for i in range(max_index):
                    row = place(fill, calc_initial(i, 0), row)
        else:
            timestr = '{:>{}}'.format(ftime(tickt).strip() + ' ', timecolsz)
            row = timestr + row

        if not isinstance(tickt, str) and tickt:
            for i in final_i[tickt]:
                if not i:
                    continue
                filltime = ' ' + ftime(tickt).strip() + ' '
                row = place(
                    MGRAY + filltime + RESET, calc_initial(i, -(len(filltime) + 1)), row
                )

        if callable(iterable_or_fn):
            iterable = map(iterable_or_fn, range(table_width))
        else:
            iterable = iterable_or_fn

        for i, values in enumerate(iterable):
            if not isinstance(values, tuple):
                text = values
                initial = 0
            elif len(values) == 2:
                initial, text = values
            elif len(values) == 3:
                i, initial, text = values
            if isinstance(text, str):
                if i < table_width:
                    offset = calc_initial(i, initial)
                    text = bshorten(text, outer_width - offset)
                    row = place(text, offset, row)
        return row

    # assemble date headers
    def date_header(i):
        thisdate = weekstart + t(days=i)
        datestr = dtime(thisdate)
        dcolor = LGRAY
        if thisdate == agendamaker.now.date():
            datestr = f'> {datestr} <'
            dcolor = WBOLD
        datestr = '{:^{}}'.format(datestr, inner_width)
        return dcolor + datestr + RESET

    newtable.append(assemble_row(None, date_header))

    # assemble full-day events
    maxslots = max(map(len, daycols))
    newtable.append(do_row(None, DASH, *CORNERS[1]))
    for j in range(maxslots):
        newtable.append(assemble_row(None, lambda i: daycols[i][j]))

    # assemble timeblocks
    newtable.append(do_row(DASH, DASH, CORNERS[1][1], *CORNERS[1][1:]))
    for tickt in agendamaker._intervals(
        contents, start_min=(True if max_start is None else max_start)
    ):
        row = None
        if has_todate and tickt == nowtick:
            nowtime = ftime(agendamaker.now, now=True).strip()
            nowtime_centered = f'{nowtime:^{inner_width}}'
            nowdex = max(len(nowtime_centered) - len(nowtime_centered.lstrip()) - 1, 0)
            pipeline = (
                LGRAY + PIPE + NOWLINE_COLOR + DASH * (inner_width + 1) * table_width
            )
            pipeline = pipeline[:-1] + LGRAY + PIPE
            pipeline = place(
                NOWLINE_COLOR + f' {nowtime} ',
                (inner_width + 1) * todate_offset + nowdex + 1,
                pipeline,
            )
            row = pipeline + RESET
        newtable.append(assemble_row(timecol[tickt], contents[tickt], row))

    return newtable


def load_evts(*args, **kwargs):
    obj = gcal.load_evts(*args, **kwargs)
    if kwargs.get('partial', False):
        return obj
    evt2short = make_evt2short(obj)
    return obj, evt2short


def parse_args(args=None):
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument(
        '-4', '--force-ipv4', action='store_true', help="force IPv4 sockets"
    )
    parser.add_argument(
        '-d',
        '--download-only',
        action='store_true',
        help="don't print anything, just refresh the calendar cache",
    )
    parser.add_argument(
        '-D',
        '--download-cal',
        action='store',
        metavar='ID',
        help="don't print anything, just download the named calendar",
    )
    parser.add_argument(
        '-W', '--width', action='store', type=int, help='set terminal width'
    )
    parser.add_argument(
        '-H', '--height', action='store', type=int, help='set terminal height'
    )
    parser.add_argument(
        '-c',
        '--calendar',
        metavar='CALENDAR',
        action='append',
        help='restrict to specified calendar(s)',
    )
    parser.add_argument(
        '-C',
        '--list-known-calendars',
        action='store_true',
        help='list all known calendars',
    )
    parser.add_argument(
        '-i',
        '--interval',
        metavar='MINUTES',
        action='store',
        type=int,
        help='interval for default/week view',
    )
    parser.add_argument(
        '-l', '--list-calendar', action='store_true', help='print a list of events'
    )
    parser.add_argument(
        '-R',
        '--no-recurring',
        action='store_true',
        help='do not print recurring events in list',
    )
    parser.add_argument(
        '-x',
        '--four-week',
        metavar='N',
        nargs='?',
        const=4,
        help='print an N-week diagram (default 4)',
    )
    parser.add_argument(
        '-X',
        '--custom-days',
        metavar='N',
        action='store',
        type=int,
        help='print an N-day diagram',
    )
    parser.add_argument(
        '-m', '--month-view', action='store_true', help='print a month diagram'
    )
    parser.add_argument(
        '-0',
        '--zero-offset',
        action='store_true',
        help='start the four-week diagram on the current day instead of Sunday',
    )
    parser.add_argument(
        '-w',
        '--week-view',
        metavar='N',
        nargs='?',
        const=0,
        help='print a multi-day view (of N days)',
    )
    parser.add_argument(
        '-s',
        '--min-start',
        metavar='N',
        action='store',
        help="start the week's day not before this time",
    )
    parser.add_argument(
        '-S',
        '--start-time',
        metavar='N',
        action='store',
        help="start the week's day at this time",
    )
    parser.add_argument('date', nargs='*', help='use this date instead of today')
    args, remain = parser.parse_known_args(args)
    if remain:
        raise Exception('unrecognized arguments: {}'.format(' '.join(remain)))

    if args.force_ipv4:
        ipv4_monkey_patch()

    if args.download_only:
        gcal.download_evts()
        print('loaded ok:', datetime.now())
        exit(0)
    elif args.download_cal:
        gcal.download_evts(args.download_cal)
        print('loaded ok:', datetime.now())
        exit(0)

    termsize = TerminalSize(args.width, args.height, False, False)
    if termsize[0] is None or termsize[1] is None:
        try:
            term_dimensions = os.get_terminal_size()
            if termsize.columns is None:
                termsize = termsize._replace(
                    columns=term_dimensions.columns, columns_auto=True
                )
            if termsize.lines is None:
                termsize = termsize._replace(
                    lines=term_dimensions.lines - 1, lines_auto=True
                )
        except OSError:
            pass

    if args.four_week is not None:
        try:
            args.four_week = int(args.four_week)
        except ValueError:
            args.date.insert(0, args.four_week)
            args.four_week = 4

    if args.week_view is not None:
        try:
            args.week_view = int(args.week_view)
        except ValueError:
            args.date.insert(0, args.week_view)
            args.week_view = 0

    modes = 0
    modes += args.list_calendar
    modes += args.four_week is not None
    modes += args.month_view
    modes += args.week_view is not None
    if modes > 1:
        return ["error: cannot specify more than one major mode"]

    if args.date:
        import parsedatetime

        pdt = parsedatetime.Calendar()
        aday = datetime(*pdt.parse(' '.join(args.date))[0][:6]).date()
        forced = True
    else:
        aday = date.today()
        forced = False

    objs = load_evts()

    if args.list_known_calendars:
        table = [
            f"{key}\t{value}"
            for key, value in gcal.get_visible_cals(objs[0]['calendars']).items()
        ]
    elif args.list_calendar:
        table = listcal(
            aday,
            args.calendar,
            no_recurring=args.no_recurring,
            forced=forced,
            objs=objs,
        )
    elif args.four_week is not None:
        table = fourweek(
            aday,
            args.calendar,
            termsize=termsize,
            zero_offset=args.zero_offset,
            table_height=args.four_week,
            no_recurring=args.no_recurring,
            objs=objs,
        )
    elif args.custom_days is not None:
        table = fourweek(
            aday,
            args.calendar,
            termsize=termsize,
            zero_offset=args.zero_offset,
            table_cells=args.custom_days,
            no_recurring=args.no_recurring,
            objs=objs,
        )
    elif args.month_view:
        aday = date(aday.year, aday.month, 1)
        ncells = calendar.monthrange(aday.year, aday.month)[1]
        table = fourweek(
            aday,
            args.calendar,
            termsize=termsize,
            zero_offset=args.zero_offset,
            table_cells=ncells,
            no_recurring=args.no_recurring,
            objs=objs,
        )
    elif args.week_view is not None:
        if args.start_time is not None:
            args.start_time = tuple(map(int, args.start_time.split(':')))
            if args.min_start is None:
                args.min_start = args.start_time
            else:
                raise Exception("can't specify both min start and start time")
        if isinstance(args.min_start, str):
            args.min_start = map(int, args.min_start.split(':'))

        table = weekview(
            aday,
            args.week_view,
            args.calendar,
            termsize=termsize,
            dark_recurring=args.no_recurring,
            zero_offset=args.zero_offset,
            interval=args.interval,
            min_start=args.min_start,
            max_start=args.start_time,
            objs=objs,
        )
    else:
        agendamaker = Agenda(args.calendar, objs=objs, interval=args.interval)
        cols = agendamaker.agenda_table(aday)
        table = agendamaker.dayview(cols, forced=forced)
        if not forced and not agendamaker.has_later:
            aday = date.today() + t(days=1)
            cols = agendamaker.agenda_table(aday, print_warning=False)
            table += agendamaker.dayview(cols)
    return table


if __name__ == '__main__':
    table = parse_args()
    Agenda.print_table(table)

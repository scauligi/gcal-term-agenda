#!/usr/bin/env python3

import os
import re
import sys
import yaml
import pickle
import sqlite3
import argparse
import functools

from collections import Counter, OrderedDict, defaultdict as ddict, namedtuple
from datetime import date, datetime, time, timedelta as t, timezone
from dateutil.tz import tzlocal
from itertools import zip_longest
from unittest.mock import patch as mock_patch, MagicMock

import gcal
from gcal import Event

from colortrans import rgb2short, short2rgb

from doublebuffer import tokenize

TerminalSize = namedtuple('TerminalSize', ['columns', 'lines'])

# https://stackoverflow.com/a/43950235
# Monkey patch to force IPv4
def ipv4_monkey_patch():
    import socket
    orig_gai = socket.getaddrinfo
    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return orig_gai(host, port, family=socket.AF_INET, type=type, proto=proto, flags=flags)
    socket.getaddrinfo = getaddrinfo

# https://stackoverflow.com/a/56842689
class reversor:
    def __init__(self, obj):
        self.obj = obj
    def __eq__(self, other):
        return other.obj == self.obj
    def __lt__(self, other):
        return other.obj < self.obj

def fg(short):
    return f'\033[38;5;{short}m'
LGRAY = fg(250)
DGRAY = fg(235)
RESET = '\033[0m'

def blen(line):
    return len(re.sub('\033.*?m', '', line))

def shorten(text, inner_width):
    if len(text) > inner_width:
        text = text[:inner_width-1] + '⋯'
    return f'{text:<{inner_width}}'

def bshorten(text, max_width):
    if max_width is None:
        max_width = min_width
    text = text.rstrip()
    l = blen(text)
    if l > max_width:
        tokens = tokenize(text)
        while l > max_width - 1:
            if not tokens.pop().startswith('\033'):
                l -= 1
        tokens.append('⋯')
        return ''.join(tokens) + RESET
    return text

# overlay `text` onto `row` at index `offset`
def place(text, offset, row):
    l = blen(row)
    if l <= offset:
        row += ' ' * (offset - l)
        return row + text
    elif offset + blen(text) < l:
        tokens = tokenize(row)
        aftertokens = []
        lastcode = None
        while l > offset:
            tok = tokens.pop()
            aftertokens.append(tok)
            if not tok.startswith('\033'):
                l -= 1
        if lastcode is None:
            for tok in reversed(tokens):
                if tok.startswith('\033'):
                    lastcode = tok
                    break
            else:
                lastcode = RESET
        l = blen(text)
        while l > 0:
            tok = aftertokens.pop()
            if not tok.startswith('\033'):
                l -= 1
            else:
                lastcode = tok
        return ''.join(tokens) + RESET + text + lastcode + ''.join(reversed(aftertokens))
    else:
        tokens = tokenize(row)
        while l > offset:
            if not tokens.pop().startswith('\033'):
                l -= 1
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

def tdtime(td):
    days = td.days
    minutes = td.seconds // 60
    hours, minutes = divmod(minutes, 60)
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

def as_date(date_or_datetime, endtime=False):
    if isinstance(date_or_datetime, datetime):
        return date_or_datetime.date()
    return date_or_datetime - t(days=int(endtime))

def as_datetime(date_or_datetime):
    if isinstance(date_or_datetime, date) and not isinstance(date_or_datetime, datetime):
        return datetime.combine(date_or_datetime, time(0), tzlocal())
    return date_or_datetime

# this exists purely because pyyaml is stupid
# and parses datetimes back as naive (utc-based) datetimes
tzl = tzlocal()
def localize(dt):
    global tzl
    if isinstance(dt, datetime):
        if not dt.tzinfo:
            return tzl.fromutc(dt)
    elif isinstance(dt, date):
        return dt
    return dt

def get_visible_cals(cals):
    try:
        allCals = []
        with open('calendars.yaml') as f:
            data = yaml.full_load(f)
            for entry in data:
                allCals.append(next(iter(entry.items())))
        allCals = OrderedDict(allCals)
    except FileNotFoundError:
        selectedCals = filter(lambda cal: 'selected' in cal and cal['selected'], cals)
        selectedCals = sorted(selectedCals, key=lambda cal: '' if 'primary' in cal and cal['primary'] else cal['summary'])
        allCals = OrderedDict()
        data = []
        for cal in selectedCals:
            key = cal['summary'].split()[0].lower()
            key = filter(str.isalpha, key)
            key = ''.join(key)
            allCals[key] = cal['id']
            data.append({key: cal['id']})
        with open('calendars.yaml', 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    allNames = list(allCals.keys())
    allCals['all'] = allNames
    return allCals

def filter_calendars(obj, calendars):
    callist = []
    allCals = get_visible_cals(obj['calendars'])
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

def _ev_entry(ev):
    return {
        'id': ev.id,
        'start': ev.start,
        'end': ev.end,
        'summary': ev.summary,
        'location': ev.location,
        'startdate': as_date(ev.start),
        'enddate': as_date(ev.end, endtime=True),
        'cancelled': ev.cancelled,
        'calendar': ev.calendar,
        'hastime': isinstance(ev.start, datetime),
        'blob': pickle.dumps(ev, protocol=-1),
    }

def download_evts():
    db = sqlite3.connect('evts.sqlite3')
    with mock_patch('pickle.dumps'):
        keys = list(_ev_entry(MagicMock()).keys())
        db.execute(f'create table if not exists events (id PRIMARY KEY,{",".join(keys[1:])})')
    now = datetime.now(tzlocal())
    tries_remaining = 2
    while tries_remaining:
        try:
            tries_remaining -= 1
            cals = gcal.s().calendarList().list().execute()['items']
            break
        except ConnectionResetError:
            if not tries_remaining:
                raise
            gcal.load_http_auth()
    calmap = {cal['id']: cal for cal in cals}
    allCals = get_visible_cals(cals)
    try:
        old_obj = load_evts(print_warning=False, partial=True)
        syncTokens = {cal['id']: cal.get('syncToken') for cal in old_obj['calendars']}
    except FileNotFoundError:
        syncTokens = {}
    for calId in allCals.values():
        if not isinstance(calId, str) or '@' not in calId:
            continue
        print(f'downloading {calId}...')
        kwargs = {
            'calendarId': calId,
            'singleEvents': True,
            'maxResults': 2500,
            'syncToken': syncTokens.get(calId),
        }
        pagenum = 0
        while True:
            pagenum += 1
            if pagenum > 1:
                print(f'  downloading page {pagenum}...')
            try:
                r = gcal.s().events().list(**kwargs).execute()
            except gcal.HttpError as e:
                if int(e.resp['status']) == 410:
                    print("  410'd, redownloading...")
                    db.execute(f'''delete from events where calendar = ?''', (calId,))
                    if calId in syncTokens:
                        del syncTokens[calId]
                    if 'syncToken' in kwargs:
                        del kwargs['syncToken']
                    if 'pageToken' in kwargs:
                        del kwargs['pageToken']
                    pagenum = 0
                    continue
                else:
                    raise
            except Exception as e:
                print(repr(e))
            entries = []
            deleting = []
            for e in r['items']:
                if e['status'] == 'cancelled':
                    deleting.append((e['id'],))
                else:
                    ev = Event.unpkg(e)
                    ev.calendar = calId
                    entries.append(_ev_entry(ev))
            db.executemany(f'''insert into events values ({",".join(f":{key}" for key in keys)})
                    on conflict(id) do update set {",".join(f"{key}=:{key}" for key in keys[1:])}''', entries)
            db.executemany(f'''delete from events where id = ?''', deleting)
            if 'nextPageToken' in r:
                kwargs['pageToken'] = r['nextPageToken']
                continue
            if 'nextSyncToken' in r:
                calmap[calId]['syncToken'] = r['nextSyncToken']
            break
        db.commit()
    obj = {
        'calendars': cals,
        'timestamp': now,
    }
    with open('evts.yaml', 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
    with open('evts.pickle', 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
    return obj

def string_outofdate(obj, now=None):
    if not now:
        now = datetime.now(tzlocal())
    if obj['timestamp'] + t(minutes=6) < now:
        return fg(3) + 'Warning: timestamp is out of date by ' + tdtime(now - obj['timestamp']) + RESET
    return None

def load_evts(*, print_warning=False, partial=False):
    now = datetime.now(tzlocal())
    try:
        try:
            with open('evts.pickle', 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            with open('evts.yaml') as f:
                obj = yaml.full_load(f)
            obj['timestamp'] = localize(obj['timestamp'])
            # for evt in obj['events']:
            #     evt.start = localize(evt.start)
            #     evt.end = localize(evt.end)
        if obj['timestamp'] > now:
            print(obj['timestamp'])
        elif obj['timestamp'] + t(minutes=5) < now:
            if print_warning:
                print('timestamp out of date', file=sys.stderr)
    except FileNotFoundError:
        raise
    if partial:
        return obj

    obj['db'] = sqlite3.connect('file:evts.sqlite3?mode=ro', uri=True)

    evt2short = make_evt2short(obj)
    return obj, evt2short

def make_evt2short(obj):
    cal2short = {}
    cal2dark = {}
    for cal in obj['calendars']:
        code = cal['backgroundColor']
        rgb = [int(code[x:x+2], 16) for x in (1, 3, 5)]
        v = max(rgb)
        new_v = max(v - 70, 0)
        scaling = new_v / v
        dark_code = ''.join(f'{round(x*scaling):02x}' for x in rgb)
        cal2short[cal['id']] = rgb2short(code)[0]
        cal2dark[cal['id']] = rgb2short(dark_code)[0]
    def evt2short(evt, dark=False):
        if dark:
            return cal2dark[evt.calendar]
        return cal2short[evt.calendar]
    return evt2short

def get_events(obj, todate, ndays, callist):
    events = []
    if ndays >= 0:
        todates = (todate, todate, ndays)
        ndays_clause = 'and date(startdate) < date(?, "+" || ? || " days")'
    else:
        todates = (todate,)
        ndays_clause = ''
    callist = list(set(callist))
    rows = obj['db'].execute(f"""
          select blob from events
          where
                date(enddate) >= date(?)
                 {ndays_clause}
             and not cancelled
             and calendar in ({','.join("?"*len(callist))})
            order by startdate, hastime, datetime(start)
    """, (*todates,
          *callist))
    return [pickle.loads(row[0]) for row in rows]

class Agenda:
    def __init__(self, calendars, objs=None, dark_recurring=False, interval=None):
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

    # assumes times with granularity at minutes
    def quantize(self, thetime, endtime=False):
        if endtime:
            return self.quantize(thetime - t(minutes=1))
        minutes = self.interval.seconds // 60
        theminutes = (thetime.hour * 60 + thetime.minute) // minutes * minutes
        return datetime(thetime.year, thetime.month, thetime.day, theminutes // 60, theminutes % 60, tzinfo=thetime.tzinfo)

    def agenda_table(self, todate, ndays=None, print_warning=True):
        self.todate = todate
        self.has_later = False

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

        events = get_events(self.obj, todate, actual_ndays, self.callist)

        for evt in events:
            # get column (1-indexed)
            # accounts for events that start before the first day
            index = max((as_date(evt.start) - todate).days, 0) + 1
            if not isinstance(evt.start, datetime):
                # full-day event
                table[index][None].append(evt)
                continue
            # timeblock event
            tickt = self.quantize(evt.start).time()
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
            minutes = tickt.hour * 60 + tickt.minute
        while tickt <= max(max(col.keys(), default=time()) for col in cols):
            yield tickt
            minutes += self.interval.seconds // 60
            tickt = time(minutes // 60, minutes % 60)

    # convenience for looping until an end time
    def _until(self, starttick, endtick):
        tick = starttick
        while tick < endtick:
            yield tick
            tick += self.interval

    def _evtcol(self, *evtcols, forced, nowtick=None):
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
                tick = datetime.combine(self.todate + t(days=col_index), tickt, tzlocal())
                tock = tick + self.interval

                for evt in evtcol[tickt]:
                    expand = _expand(evt)
                    summary = evt.summary
                    endtext = ''

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
                            endtext = ' ' + evt.location + endtext

                    summary = fg(self.evt2short(evt)) + summary + endtext + RESET + '   '

                    # place into leftmost region that is large enough
                    prev_end = 0
                    for i, (icol_index, initial, text) in enumerate(sorted(contents[tickt])):
                        if icol_index != col_index:
                            continue
                        if initial - prev_end >= blen(summary):
                            break
                        prev_end = initial + blen(text)
                    contents[tickt].append((col_index, prev_end, summary))
                    initial = prev_end

                    # drop anchor on long events
                    if expand:
                        lasttick = self.quantize(evt.end, endtime=True)
                        for endtick in self._until(tock, evt.end):
                            endtock = endtick + self.interval
                            if evt.end == endtock:
                                text = '_|_ '
                            elif evt.end < endtock:
                                text = '-+- ({}) '.format(ftime(evt.end).strip())
                            else:
                                text = ' |  '
                            if endtick == endtock:
                                text += evt.location + '  '
                            text = fg(self.evt2short(evt)) + text + RESET
                            contents[endtick.time()].append((col_index, initial, text))
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

        # if there are no actual events
        if not any(evtcol.items()):
            newtable.append(f'{dtime(self.todate)} {LGRAY}{timefield}  no events{RESET}')
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
                span = ' ({}->{})'.format(startspan, endspan)
            summary = fg(self.evt2short(evt)) + evt.summary + RESET
            newtable.append([timefield, summary + span])

        # whether to do "now" arrow
        is_todate = self.now.date() == self.todate

        contents = self._evtcol(evtcol, forced=forced, nowtick=nowtick)

        lasttick = max(contents.keys(), default=time())
        if is_todate:
            # make sure there's a ddict key entry for nowtick
            # so that self._intervals iterates at least that far
            contents[nowtick.time()]

        # assemble newtable from contents
        for tickt in self._intervals(contents):
            tick = datetime.combine(self.todate, tickt, tzlocal())

            # skip blank slots until the first event
            if not did_first and not timecol[tickt] and not (is_todate and tick == nowtick):
                continue
            # skip blank slots after the last event
            if tickt > lasttick and not (is_todate and tick == nowtick):
                continue

            # print tick time for event starts
            timefield = ftime(tick if timecol[tickt] else None)

            # print each event at the right output column
            content = ''
            prev_end = 0
            for (_, initial, text) in contents[tickt]:
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
        newtable = [f'{line[0]} {line[1]}  {line[2]}' for line in newtable]

        # collect status messages
        if timecol[None]:
            newtable = timecol[None] + newtable

        return newtable

    @staticmethod
    def print_table(table):
        lines = '\n'.join(line.rstrip() for line in table)
        lines = re.sub(r'\n{8,}', r'\n'*8, lines)
        lines = re.sub(r'\n*$', '', lines)
        print(lines)


def listcal(todate, calendars, no_recurring=False, forced=False, objs=None):
    today = datetime.combine(todate, time(0), tzlocal())
    now = datetime.now(tzlocal())
    obj, evt2short = objs

    table = []
    if outofdate := string_outofdate(obj, now):
        table.append(outofdate)

    callist = filter_calendars(obj, calendars)

    weekday = todate.isoweekday() % 7
    nextsunday = 7 - weekday
    nextmonth = date(today.year + today.month // 12, today.month % 12 + 1, 1)
    nextmonth = datetime.combine(nextmonth, time(0), tzlocal())
    follmonth = date(today.year + (today.month + 1) // 12, (today.month + 1) % 12 + 1, 1)
    follmonth = datetime.combine(follmonth, time(0), tzlocal())

    highwater = [
        (None,                              '== ONGOING =='),
        (today,                             '== TODAY =='),
        (today + t(days=1),                 '== TOMORROW =='),
        (today + t(days=2),                 '== THIS WEEK =='),
        (today + t(days=nextsunday),        '== NEXT WEEK =='),
        (today + t(days=(nextsunday + 7)),  '== FOLLOWING WEEK =='),
        (today + t(days=(nextsunday + 14)), '== THIS MONTH =='),
        (nextmonth,                         '== NEXT MONTH =='),
        (follmonth,                         '== THE FUTURE =='),
    ]

    events = get_events(obj, todate, 1 if forced else -1, callist)

    seen = Counter()
    for evt in events:
        if no_recurring and evt.recurring:
            continue
        start = as_datetime(evt.start)
        end = as_datetime(evt.end)
        seen[evt.uid] += 1
        if seen[evt.uid] > 1:
            continue
        s = None
        while highwater and (highwater[0][0] is None or start >= highwater[0][0]):
            _, s = highwater.pop(0)
        if s:
            table.append(fg(4) + s + RESET)
        dark = isinstance(evt.end, datetime) and evt.end < now
        white = fg(8) if dark else ''
        fmt = '  '
        fmt += white + dtime(evt.start) + RESET
        fmt += ' '
        fmt += white + ftime(evt.start if isinstance(evt.start, datetime) else None)
        fmt += ' '
        fmt += fg(evt2short(evt, dark=dark)) + evt.summary + RESET
        fmt += ' '
        fmt += white + tdtime(evt.end - evt.start) + RESET
        table.append(fmt)
    return table


CORNERS = """\
┌┬┐▄
├┼┤█
└┴┘▀""".split('\n')
DASH = "─"
PIPE = "│"
THICK = "█"

def fourweek(todate, calendars, termsize=None, objs=None, zero_offset=False):
    table_width = 7
    table_height = 4

    inner_width = (termsize.columns - (table_width + 1)) // table_width
    inner_height = ((termsize.lines - 1) - (table_height + 1)) // table_height

    table = []

    today = datetime.combine(todate, time(0), tzlocal())
    offset = todate.isoweekday() % 7
    rev_offset = 0
    if zero_offset:
        rev_offset = (7 - offset) % 7
        offset = 0

    def do_row(fill, left, mid=None, right=None, thick=None):
        if mid is None:
            mid = left
        if right is None:
            right = left
        if thick is None:
            thick = mid
        line = left
        for _ in range(table_width):
            line += fill * inner_width + mid
        if rev_offset:
            index = rev_offset * (inner_width + 1)
            line = line[:index] + thick + line[index+1:]
        line = line[:-1] + right
        return LGRAY + line + RESET

    # set up table borders
    table.append(do_row(DASH, *CORNERS[0]))
    for _ in range(table_height):
        for _ in range(inner_height):
            table.append([])
        table.append(do_row(DASH, *CORNERS[1]))
    table.pop()
    table.append(do_row(DASH, *CORNERS[2]))

    obj, _evt2short = objs
    now = datetime.now(tzlocal())
    nowdate = now.date()
    def evt2short(evt):
        if as_date(evt.start) > as_date(now):
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

    cells = [(list(), list()) for _ in range(table_width * table_height)]

    calstart = todate - t(days=offset)
    events = get_events(obj, todate - t(days=offset), 4 * 7, callist)

    for evt in events:
        start = as_datetime(evt.start)
        end = as_datetime(evt.end)
        cellnum = (start.date() - calstart).days
        if cellnum in range(0, len(cells)):
            if isinstance(evt.start, datetime):
                text = ftime(evt.start) + ' ' + evt.summary
            else:
                text = ' ' + evt.summary
            text = shorten(text, inner_width)
            text = fg(evt2short(evt)) + text + RESET
            cells[cellnum][choice(evt)].append(text)
        if not isinstance(evt.start, datetime) and (end - start).days > 1:
            for day in range(1, (evt.end - evt.start).days):
                cellnum = (start.date() - calstart).days + day
                if cellnum in range(0, len(cells)):
                    text = '> ' + evt.summary
                    text = shorten(text, inner_width)
                    text = fg(evt2short(evt)) + text + RESET
                    cells[cellnum][choice(evt)].append(text)

    # overwrite table with content of cells
    for i in range(table_height):
        for j in range(table_width):
            cell, cell_recurring = cells[i * table_width + j]
            if cell:
                cell.append(' ' * inner_width)
            cell.extend(cell_recurring)
            for l in range(inner_height):
                lineIndex = i * (inner_height + 1) + l + 1
                text = ' ' * inner_width
                if l == 0:
                    celldate = todate + t(days=(i * table_width + j - offset))
                    datetext = dtime(celldate)
                    dcolor = LGRAY
                    if celldate == now.date():
                        datetext = f'> {datetext} <'
                        dcolor = RESET
                    text = dcolor + shorten(f'{datetext:^{inner_width}}', inner_width) + RESET
                elif l == 1:
                    text = LGRAY + DASH * inner_width + RESET
                else:
                    l -= 2
                    if l + 2 == inner_height - 1 and len(cell) > l + 1:
                        text = shorten(format('... more ...', f'^{inner_width}'), inner_width)
                        text = fg(4) + text + RESET
                    elif l < len(cell):
                        text = cell[l]
                table[lineIndex].append(text)

    newtable = []
    for line in table:
        if isinstance(line, list):
            text = ''
            for i, segment in enumerate(line):
                if rev_offset and rev_offset == i:
                    text += LGRAY + THICK + RESET
                else:
                    text += LGRAY + PIPE + RESET
                text += segment
            text += LGRAY + PIPE + RESET
            newtable.append(text)
        else:
            newtable.append(line)
    if outofdate := string_outofdate(obj, now):
        newtable[0] = place(outofdate, 2, newtable[0])
    return newtable

def weekview(todate, week_ndays, calendars, termsize=None, objs=None, dark_recurring=False, zero_offset=False, interval=None):
    now = datetime.now(tzlocal())
    table_width = week_ndays if week_ndays > 0 else 7
    interval = interval or 30

    offset = 0
    if week_ndays == 0:
        offset = todate.isoweekday() % 7
        if zero_offset:
            offset = (offset - 1) % 7
        week_ndays = 7
    weekstart = todate - t(days=offset)

    agendamaker = Agenda(calendars, objs=objs, dark_recurring=dark_recurring, interval=interval)
    timecol, *evtcols = agendamaker.agenda_table(weekstart, ndays=table_width)

    timecolsz = len(ftime()) + 2
    inner_width = (termsize.columns - timecolsz - (table_width + 1)) // table_width

    # full-day events part 1
    OPEN = object()
    CLOSED = object()
    daycols = [ddict(lambda: OPEN) for evtcol in evtcols]
    for i, evtcol in enumerate(evtcols):
        for evt in evtcol.pop(None, []):
            ndays = (evt.end - weekstart).days - i
            subcols = daycols[i:i+ndays]
            topslot = max(map(len, subcols))
            for j in range(topslot):
                if all(j >= len(daycol) or daycol[j] is OPEN for daycol in subcols):
                    # found an empty slot range
                    break
            else:
                j = topslot

            summary = ' ' + evt.summary
            if evt.start < weekstart:
                summary = '⋯' + DASH * timecolsz + summary
            if ndays > 1:
                outlen = ndays * (inner_width + 1) - 1
                summary += ' ' + DASH * (outlen - len(evt.summary) - 3) + '>'

            daycol = subcols[0]
            # daycol.extend([OPEN] * (j - len(daycol) + 1))
            assert daycol[j] is OPEN
            daycol[j] = fg(agendamaker.evt2short(evt)) + summary + RESET
            if evt.start < weekstart:
                daycol[j] = (-(timecolsz + 1), daycol[j])

            for daycol in subcols[1:]:
                # daycol.extend([OPEN] * (j - len(daycol) + 1))
                assert daycol[j] is OPEN
                daycol[j] = CLOSED

    final_i = {}
    for i, evtcol in enumerate(evtcols):
        for tickt in evtcol:
            final_i[tickt] = i

    contents = agendamaker._evtcol(*evtcols, forced=True)

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
            timestr = ftime(tickt) + '  '
        return LGRAY + timestr + line + RESET

    def assemble_row(tickt, iterable_or_fn):
        def calc_initial(i, initial):
            return i * (inner_width + 1) + 1 + initial + timecolsz

        row = do_row(tickt, ' ', PIPE)
        if not isinstance(tickt, str) and tickt:
            for i in range(final_i[tickt]):
                fill = DASH * inner_width
                if i == final_i[tickt] - 1:
                    fill = fill[:-1] + '>'
                row = place(DGRAY + fill + RESET,
                            calc_initial(i, 0),
                            row)

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
                offset = calc_initial(i, initial)
                text = bshorten(text, termsize.columns - offset)
                row = place(text, offset, row)
        return row

    # assemble date headers
    def date_header(i):
        thisdate = weekstart + t(days=i)
        datestr = dtime(thisdate)
        if thisdate == now.date():
            datestr = f'> {datestr} <'
        datestr = '{:^{}}'.format(datestr, inner_width)
        return LGRAY + datestr + RESET
    newtable.append(assemble_row(None, date_header))

    # assemble full-day events
    maxslots = max(map(len, daycols))
    if maxslots:
        newtable.append(do_row(None, DASH, *CORNERS[1]))
    for j in range(maxslots):
        newtable.append(assemble_row(None, lambda i: daycols[i][j]))


    # assemble timeblocks
    newtable.append(do_row(DASH, DASH, CORNERS[1][1], *CORNERS[1][1:]))
    for tickt in agendamaker._intervals(contents, start_min=True):
        newtable.append(assemble_row(timecol[tickt], contents[tickt]))

    return newtable

SOCK = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unix_sock')
import asyncio
import signal
from async_utils import read_pickled, write_pickled
def server():
    if os.path.exists(SOCK):
        # not a race since we check again later
        raise Exception(SOCK + ' already exists')
    obj_lock = asyncio.Lock()
    objs = load_evts(print_warning=True)
    db = objs[0]['db']

    async def download_loop():
        nonlocal objs
        nonlocal obj_lock
        while True:
            try:
                print(datetime.now(), 'downloading...')
                async with obj_lock:
                    obj = download_evts()
                    obj['db'] = db
                    evt2short = make_evt2short(obj)
                    objs = (obj, evt2short)
                print(datetime.now(), 'loaded ok')
            except Exception as e:
                print('download loop:', e)
            await asyncio.sleep(5*60)
    async def handle_connection(reader, writer):
        nonlocal objs
        nonlocal obj_lock
        argv, termsize = await read_pickled(reader)
        print(datetime.now(), 'serving client:', argv)
        try:
            async with obj_lock:
                table = parse_args(argv, termsize, objs=objs)
        except Exception as e:
            table = [str(e)]
        write_pickled(writer, table)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    async def start_server():
        timer = asyncio.create_task(download_loop())
        if os.path.exists(SOCK):
            # XXX could race
            raise Exception(SOCK + ' already exists')
        server = await asyncio.start_unix_server(handle_connection, SOCK)
        # XXX could race
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
        try:
            async with server:
                await server.serve_forever()
        finally:
            cleanup()
    def cleanup(_signum=None, _frame=None):
        try:
            os.remove(SOCK)
        except:
            pass
        exit(1)
    asyncio.run(start_server())
def client(argv, termsize):
    async def do_client():
        reader, writer = await asyncio.open_unix_connection(SOCK)
        write_pickled(writer, (argv, termsize))
        await writer.drain()
        table = await read_pickled(reader)
        writer.close()
        await writer.wait_closed()
        return table
    return asyncio.run(do_client())

def parse_args(argv, termsize, objs=None):
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument('-f', '--force-download-check', action='store_true', help='overrides -n')
    parser.add_argument('-c', '--calendar', metavar='CALENDAR', action='append', help='restrict to specified calendar(s)')
    parser.add_argument('-i', '--interval', metavar='MINUTES', action='store', type=int, help='interval for default/week view')
    parser.add_argument('-l', '--list-calendar', action='store_true', help='print a list of events')
    parser.add_argument('-R', '--no-recurring', action='store_true', help='do not print recurring events in list')
    parser.add_argument('-x', '--four-week', action='store_true', help='print a four-week diagram')
    parser.add_argument('-0', '--zero-offset', action='store_true', help='start the four-week diagram on the current day instead of Sunday')
    parser.add_argument('-w', '--week-view', metavar='N', nargs='?', const=0, type=int, help='print a multi-day view (of N days)')
    parser.add_argument('date', nargs='*', help='use this date instead of today')
    args, remain = parser.parse_known_args(argv)
    if remain:
        raise Exception('unrecognized arguments: {}'.format(' '.join(remain)))

    modes = 0
    modes += args.list_calendar
    modes += args.four_week
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

    if objs is None:
        objs = load_evts()

    if args.list_calendar:
        table = listcal(aday, args.calendar, no_recurring=args.no_recurring, forced=forced, objs=objs)
    elif args.four_week:
        table = fourweek(aday, args.calendar, termsize=termsize, zero_offset=args.zero_offset, objs=objs)
    elif args.week_view is not None:
        table = weekview(aday, args.week_view, args.calendar, termsize=termsize, dark_recurring=args.no_recurring, zero_offset=args.zero_offset, interval=args.interval, objs=objs)
    else:
        agendamaker = Agenda(args.calendar, objs=objs, interval=args.interval)
        cols = agendamaker.agenda_table(aday)
        table = agendamaker.dayview(cols, forced=forced)
        if not forced and not agendamaker.has_later:
            aday = date.today() + t(days=1)
            cols = agendamaker.agenda_table(aday, print_warning=False)
            table += agendamaker.dayview(cols)
    return table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-4', '--force-ipv4', action='store_true', help="force IPv4 sockets")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--download-only', action='store_true', help="don't print anything, just refresh the calendar cache")
    group.add_argument('--server', action='store_true', help="start server")
    group.add_argument('--client', action='store_true', help="run as client")
    parser.add_argument('-W', '--width', action='store', type=int, help='set terminal width')
    parser.add_argument('-H', '--height', action='store', type=int, help='set terminal height')
    args, remainder = parser.parse_known_args()

    termsize = TerminalSize(args.width, args.height)
    if termsize[0] is None or termsize[1] is None:
        try:
            term_dimensions = os.get_terminal_size()
            if termsize.columns is None:
                termsize = termsize._replace(columns=term_dimensions.columns)
            if termsize.lines is None:
                termsize = termsize._replace(lines=term_dimensions.lines)
        except OSError:
            pass

    if args.force_ipv4:
        ipv4_monkey_patch()

    if args.download_only:
        download_evts()
        print('loaded ok:', datetime.now())
        exit(0)
    elif args.server:
        server()
        exit(0)
    elif args.client:
        return client(remainder, termsize)
    else:
        return parse_args(remainder, termsize)

if __name__ == '__main__':
    table = main()
    Agenda.print_table(table)

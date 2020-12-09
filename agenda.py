#!/usr/bin/env python3

import os
import re
import sys
import yaml
import pickle

from collections import Counter, OrderedDict
from datetime import date, datetime, time, timedelta as t, timezone
from dateutil.tz import tzlocal

import gcal
from gcal import Event

from colortrans import rgb2short

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
RESET = '\033[0m'

def blen(line):
    return len(re.sub('\033.*?m', '', line))

def dtime(dt=None):
    if not dt:
        return re.sub(r'.', ' ', date.today().strftime('%a %m/%d'))
    s = dt.strftime('%a %m/%d')
    s = re.sub(r'0(\d)/', r' \1/', s)
    s = re.sub(r'/0(\d)$', r'/\1 ', s)
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

def as_date(date_or_datetime):
    if isinstance(date_or_datetime, datetime):
        return date_or_datetime.date()
    return date_or_datetime

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

class CurEvent:
    def __init__(self, evt, idx):
        self.evt = evt
        self.idx = idx
        self.location = evt.location
    def pop_location(self):
        location = self.location
        self.location = ''
        if location:
            location += '  '
        return location
    def __eq__(self, o):
        if type(o) == type(self.evt):
            return o == self.evt
        if not isinstance(o, CurEvent):
            return False
        return o.evt == self.evt

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
                exit(1)
            _getCal(lookup)
        else:
            callist.append(calendar)
    _getCal(calendars)
    return callist

def download_evts(today):
    now = datetime.now(tzlocal())
    evs = []
    day_evs = []
    cals = gcal.s().calendarList().list().execute()['items']
    allCals = get_visible_cals(cals)
    for calId in allCals.values():
        if not isinstance(calId, str) or '@' not in calId:
            continue
        r = gcal.s().events().list(calendarId=calId,
                singleEvents=True,
                orderBy='startTime',
                timeMin=(today - t(days=10)).isoformat(),
                timeMax=(today + t(days=30)).isoformat()).execute()
        for e in r['items']:
            ev = Event.unpkg(e)
            ev.calendar = calId
            evs.append(ev)
    evs.sort(key=lambda e: (as_datetime(e.start), reversor(as_datetime(e.end))))
    obj = {'events': evs, 'calendars': cals, 'timestamp': now}
    with open('evts.yaml', 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
    with open('evts.pickle', 'wb') as f:
        pickle.dump(obj, f)
    return obj

def string_outofdate(obj, now=None):
    if not now:
        now = datetime.now(tzlocal())
    if obj['timestamp'] + t(minutes=6) < now:
        return fg(3) + 'Warning: timestamp is out of date by ' + tdtime(now - obj['timestamp']) + RESET
    return None

def print_outofdate(obj):
    if s := string_outofdate(obj):
        print(s, file=sys.stderr)

def load_evts(today=None, no_download=False, print_warning=True):
    now = datetime.now(tzlocal())
    today = datetime.combine(today or date.today(), time(0), tzlocal())
    try:
        try:
            with open('evts.pickle', 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            with open('evts.yaml') as f:
                obj = yaml.full_load(f)
            obj['timestamp'] = localize(obj['timestamp'])
            for evt in obj['events']:
                evt.start = localize(evt.start)
                evt.end = localize(evt.end)
        if obj['timestamp'] > now:
            print(obj['timestamp'])
            exit(1)
        elif obj['timestamp'] + t(minutes=5) < now and not no_download:
            raise FileNotFoundError
    except FileNotFoundError:
        obj = download_evts(today)
    if print_warning:
        print_outofdate(obj)

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

    return obj, evt2short

class Agenda:
    def __init__(self, todate, calendars, no_download=False):
        self.now = datetime.now(tzlocal())
        self.todate = aday
        self.today = datetime.combine(self.todate, time(0), tzlocal())
        self.obj, self.evt2short = load_evts(self.today, no_download=no_download, print_warning=False)
        self.callist = filter_calendars(self.obj, calendars)

        self.interval = t(minutes=15)

        self.has_later = False

    def _commit(self):
        if self.now >= self.tick and self.now < self.tock:
            self.curline += '  <-- ' + ftime(self.now, now=True)
        if self.curline or (self.did_first and self.tick > self.now):
            if not self.timefield:
                self.timefield = ftime()
            self.table.append(self.datefield + ' ' + self.timefield + '  ' + self.curline)
            self.datefield = dtime()
            self.timefield = ''
            self.curline = ''
            self.did_first = True

    def _advance(self):
        self.tick += self.interval
        self.tock = self.tick + self.interval
        i = 0
        while i < len(self.current_events):
            curevt = self.current_events[i]
            assert curevt.evt.end >= self.tick
            if blen(self.curline) < curevt.idx:
                self.curline += ' ' * (curevt.idx - blen(self.curline))
            self.curline += fg(self.evt2short(curevt.evt))
            if curevt.evt.end == self.tock:
                self.curline += '_|_ '
                self.current_events.pop(i)
            elif curevt.evt.end < self.tock:
                self.curline += '-+- ({}) '.format(ftime(curevt.evt.end).strip())
                self.current_events.pop(i)
            else:
                self.curline += ' |  '
                i += 1
            self.curline += curevt.pop_location()
            self.curline += RESET

    def agenda_table(self):
        self.tick = self.today
        self.tock = self.tick + self.interval

        self.table = []
        self.current_events = []

        enddate = self.todate + t(days=1)

        last_date = None
        self.did_first = False
        self.datefield = ''
        self.timefield = ''
        self.curline = ''
        for evt in self.obj['events']:
            if evt.calendar not in self.callist:
                continue
            if evt.cancelled:
                continue
            start = as_date(evt.start)
            end = as_date(evt.end)
            if end < self.todate:
                continue
            elif start >= enddate:
                break
            day = as_date(evt.start)
            if day != last_date:
                self.datefield = dtime(day)
                last_date = day
            if not isinstance(evt.start, datetime):
                # kludgy bugfix
                if evt.end == self.todate:
                    continue
                dateline = evt.summary
                dateline = fg(self.evt2short(evt)) + dateline
                dateline += RESET
                self.table.append(self.datefield + ' ' + ftime().replace(' ', '-') + '  ' + dateline)
                self.datefield = dtime()
                continue
            while self.tick < as_datetime(evt.start):
                self._commit()
                self._advance()
            dark = False
            summary = ''
            if not self.timefield:
                self.timefield = ftime(evt.start)
            else:
                self.curline += '   '
                if ftime(evt.start) != self.timefield:
                    summary += ' ({})'.format(ftime(evt.start).strip())
            if evt.end <= self.tock and evt.end != evt.start:
                summary += ' (-> {})'.format(ftime(evt.end).strip())
            if evt.end > self.now and evt.end > self.tock:
                self.current_events.append(CurEvent(evt, blen(self.curline)))
                self.has_later = True
            elif evt.end > self.now:
                summary += ' ' + evt.location
            else:
                dark = True
            summary = fg(self.evt2short(evt, dark=dark)) + evt.summary + summary + RESET
            self.curline += summary
        while self.current_events:
            self._commit()
            self._advance()
        self._commit()

        if outofdate := string_outofdate(self.obj, self.now):
            self.table.insert(0, outofdate)

        return self.table


    def print_table(self, table):
        lines = re.sub(r'\n{8,}', r'\n'*8, '\n'.join(line.rstrip() for line in table)).split('\n')
        print('\n'.join(lines))


def listcal(todate, calendars, no_download=False, no_recurring=False):
    today = datetime.combine(todate, time(0), tzlocal())
    now = datetime.now(tzlocal())
    obj, evt2short = load_evts(today, no_download=no_download)

    callist = filter_calendars(obj, calendars)

    weekday = today.isoweekday() % 7
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

    seen = Counter()
    for evt in obj['events']:
        if evt.calendar not in callist:
            continue
        if evt.cancelled:
            continue
        if no_recurring and evt.recurring:
            continue
        start = as_datetime(evt.start)
        end = as_datetime(evt.end)
        if end > today:
            seen[evt.uid] += 1
            if seen[evt.uid] > 1:
                continue
            s = None
            while highwater and (highwater[0][0] is None or start >= highwater[0][0]):
                _, s = highwater.pop(0)
            if s:
                print(fg(4) + s + RESET)
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
            print(fmt)


CORNERS = """\
┌┬┐▄
├┼┤█
└┴┘▀""".split('\n')
DASH = "─"
PIPE = "│"
THICK = "█"

def fourweek(todate, calendars, no_download=False, zero_offset=False):
    termsize = os.get_terminal_size()

    table_width = 7
    table_height = 4

    inner_width = (termsize.columns - (table_width + 1)) // table_width
    inner_height = ((termsize.lines - 1) - (table_height + 1)) // table_height

    table = []

    today = datetime.combine(todate, time(0), tzlocal())
    offset = today.isoweekday() % 7
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
        return line

    # set up table borders
    table.append(do_row(DASH, *CORNERS[0]))
    for _ in range(table_height):
        for _ in range(inner_height):
            table.append([])
        table.append(do_row(DASH, *CORNERS[1]))
    table.pop()
    table.append(do_row(DASH, *CORNERS[2]))

    obj, _evt2short = load_evts(today, no_download=no_download)
    now = datetime.now(tzlocal())
    def evt2short(evt):
        if as_date(evt.start) < todate:
            dark = True
        elif as_date(evt.start) == todate:
            dark = isinstance(evt.end, datetime) and evt.end <= now
        else:
            dark = evt.recurring
        return _evt2short(evt, dark=dark)
    def choice(evt):
        if as_date(evt.start) == todate:
            return isinstance(evt.end, datetime) and evt.start >= now
        else:
            return evt.recurring

    callist = filter_calendars(obj, calendars)

    cells = [(list(), list()) for _ in range(table_width * table_height)]

    def shorten(text):
        if len(text) > inner_width:
            text = text[:inner_width-1] + '⋯'
        return f'{text:<{inner_width}}'

    for evt in obj['events']:
        if evt.calendar not in callist:
            continue
        if evt.cancelled:
            continue
        start = as_datetime(evt.start)
        end = as_datetime(evt.end)
        cellnum = (start.date() - todate).days
        cellnum += offset
        if cellnum in range(0, len(cells)):
            if isinstance(evt.start, datetime):
                text = ftime(evt.start) + ' ' + evt.summary
            else:
                text = ' ' + evt.summary
            text = shorten(text)
            text = fg(evt2short(evt)) + text + RESET
            cells[cellnum][choice(evt)].append(text)
        if not isinstance(evt.start, datetime) and (end - start).days > 1:
            for day in range(1, (evt.end - evt.start).days):
                cellnum = (start.date() - todate).days + offset + day
                if cellnum in range(0, len(cells)):
                    text = '> ' + evt.summary
                    text = shorten(text)
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
                    datetext = dtime(todate + t(days=(i * table_width + j - offset)))
                    if i * table_width + j == offset:
                        datetext = f'> {datetext} <'
                    text = shorten(f'{datetext:^{inner_width}}')
                elif l == 1:
                    text = DASH * inner_width
                else:
                    l -= 2
                    if l + 2 == inner_height - 1 and len(cell) > l + 1:
                        text = shorten(format('... more ...', f'^{inner_width}'))
                        text = fg(4) + text + RESET
                    elif l < len(cell):
                        text = cell[l]
                table[lineIndex].append(text)

    for line in table:
        if isinstance(line, list):
            text = ''
            for i, segment in enumerate(line):
                if rev_offset and rev_offset == i:
                    text += THICK
                else:
                    text += PIPE
                text += segment
            text += PIPE
            print(text)
        else:
            print(line)

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-4', '--force-ipv4', action='store_true', help="force IPv4 sockets")
    parser.add_argument('-d', '--download-loop', action='store_true', help="don't print anything, just refresh the calendar cache")
    parser.add_argument('-n', '--no-download', action='store_true', help="don't attempt to refresh the calendar cache")
    parser.add_argument('-f', '--force-download-check', action='store_true', help='overrides -n')
    parser.add_argument('-c', '--calendar', metavar='CALENDAR', action='append', help='restrict to specified calendar(s)')
    parser.add_argument('-l', '--list-calendar', action='store_true', help='print a list of events')
    parser.add_argument('-R', '--no-recurring', action='store_true', help='do not print recurring events in list')
    parser.add_argument('-x', '--four-week', action='store_true', help='print a four-week diagram')
    parser.add_argument('-0', '--zero-offset', action='store_true', help='start the four-week diagram on the current day instead of Sunday')
    parser.add_argument('date', nargs='*', help='use this date instead of today')
    args = parser.parse_args()
    no_download = args.no_download and not args.force_download_check
    if args.download_loop:
        load_evts()
        print('loaded ok:', datetime.now())
        exit(0)
    if args.date:
        import parsedatetime
        pdt = parsedatetime.Calendar()
        aday = datetime(*pdt.parse(' '.join(args.date))[0][:6]).date()
    else:
        aday = date.today()
    if args.list_calendar and args.four_week:
        print("error: cannot specify both -l and -x", file=sys.stderr)
        exit(1)
    if args.list_calendar:
        listcal(aday, args.calendar, no_download=no_download, no_recurring=args.no_recurring)
        exit(0)
    elif args.four_week:
        fourweek(aday, args.calendar, no_download=no_download, zero_offset=args.zero_offset)
        exit(0)
    agendamaker = Agenda(aday, args.calendar, no_download=no_download)
    table = agendamaker.agenda_table()
    if not agendamaker.has_later and not args.date:
        table.append(dtime() + ftime() + '     <-- ' + ftime(agendamaker.now, now=True))
        aday = date.today() + t(days=1)
        agendamaker = Agenda(aday, args.calendar, no_download=True)
        table += agendamaker.agenda_table()
    agendamaker.print_table(table)

#!/usr/bin/env python3

import os
import re
import sys
import yaml

from collections import Counter, OrderedDict
from datetime import date, datetime, time, timedelta as t, timezone
from dateutil.tz import tzlocal

import gcal
from gcal import Event

from colortrans import rgb2short

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
        return '         '
    s = dt.strftime('%a %m/%d')
    s = re.sub(r'0(\d)/', r' \1/', s)
    s = re.sub(r'/0(\d)$', r'/\1 ', s)
    return s

def ftime(dt=None, now=False):
    if not dt:
        return '      '
    s = re.sub(r'^0', ' ', dt.strftime('%I:%M%P')[:-1])
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
                timeMin=(today - t(days=30)).isoformat(),
                timeMax=(today + t(days=90)).isoformat()).execute()
        for e in r['items']:
            ev = Event.unpkg(e)
            ev.calendar = calId
            if isinstance(ev.start, datetime):
                evs.append(ev)
            else:
                day_evs.append(ev)
    evs.sort(key=lambda ev: ev.start)
    day_evs.sort(key=lambda ev: ev.start)
    obj = {'events': evs, 'day_events': day_evs, 'calendars': cals, 'timestamp': now}
    with open('evts.yaml', 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
    return obj

def load_evts(today=None, no_download=False):
    now = datetime.now(tzlocal())
    today = datetime.combine(today or date.today(), time(0), tzlocal())
    try:
        with open('evts.yaml') as f:
            obj = yaml.full_load(f)
            obj['timestamp'] = localize(obj['timestamp'])
        if obj['timestamp'] > now:
            print(obj['timestamp'])
            exit(1)
        elif obj['timestamp'] + t(minutes=5) < now and not no_download:
            raise FileNotFoundError
        else:
            if obj['timestamp'] + t(minutes=5) < now:
                print(fg(3) + 'Warning: timestamp is out of date by', tdtime(now - obj['timestamp']), RESET, file=sys.stderr)
            evs = obj['events']
            for evt in obj['events']:
                evt.start = localize(evt.start)
                evt.end = localize(evt.end)
    except FileNotFoundError:
        obj = download_evts(today)
    return obj

def agenda(clear=False, aday=None, no_download=False):
    now = datetime.now(tzlocal())
    today = datetime.combine(aday or date.today(), time(0), tzlocal())
    obj = load_evts(today, no_download=no_download)

    cal2short = {}
    for cal in obj['calendars']:
        cal2short[cal['id']] = rgb2short(cal['backgroundColor'])[0]

    tick = today
    interval = t(minutes=15)

    table = []
    lines = []
    current_events = []

    while tick.date() <= today.date() + t(days=1):
        day = tick.date()
        did_date = False
        did_first = False
        for evt in obj['events']:
            if isinstance(evt.start, date) and not isinstance(evt.start, datetime):
                datefield = ''
                curline = ''
                if evt.start <= tick.date() and evt.end > tick.date():
                    curline = evt.summary
                    curline = fg(cal2short[evt.calendar]) + curline
                    curline += RESET
                    if not datefield:
                        if not did_date:
                            datefield = dtime(day)
                        else:
                            datefield = dtime()
                    table.append(datefield + ' ' + ftime().replace(' ', '-') + '  ' + curline)
                    if not did_date:
                        did_date = True
        # increments tick/tock by interval
        while tick.date() == day:
            tock = tick + interval
            datefield = ''
            timefield = ''
            curline = ''
            for evt in current_events:
                if evt.evt.end >= tick:
                    if blen(curline) < evt.idx:
                        curline += ' ' * (evt.idx - blen(curline))
                    curline += fg(cal2short[evt.evt.calendar])
                    if evt.evt.end == tock:
                        curline += '_|_ '
                    elif evt.evt.end < tock:
                        curline += '-+- ({}) '.format(ftime(evt.evt.end).strip())
                    else:
                        curline += ' |  '
                    curline += RESET
            for evt in obj['events']:
                if isinstance(evt.start, datetime):
                    if evt.start >= tick and evt.start < tock:
                        if evt.end > now:
                            current_events.append(CurEvent(evt, blen(curline)))
                        summary = evt.summary + ' ' + evt.location
                        summary = fg(cal2short[evt.calendar]) + summary
                        if not timefield:
                            timefield = ftime(evt.start)
                        else:
                            summary = '   ' + summary
                            if current_events:
                                current_events[-1].idx += 3
                            if ftime(evt.start) != timefield:
                                summary += ' ({})'.format(ftime(evt.start).strip())
                        summary += RESET
                        curline += summary
                        if evt.end > tick and evt.end <= tock:
                            try:
                                current_events.remove(evt)
                            except ValueError:
                                pass
                            if evt.end != evt.start:
                                curline += fg(cal2short[evt.calendar])
                                curline += ' (-> {})'.format(ftime(evt.end).strip())
                                curline += RESET
                    elif evt.end > tick and evt.end <= tock:
                        try:
                            current_events.remove(evt)
                        except ValueError:
                            pass
            if now >= tick and now < tock:
                curline += '  <-- ' + ftime(now, now=True)
            if not timefield:
                timefield = ftime()
            if not datefield:
                if not did_date:
                    datefield = dtime(day)
                else:
                    datefield = dtime()
            lines.append(datefield + ' ' + timefield + '  ' + curline)
            if curline:
                if not did_first or tick <= now:
                    lines = [lines[-1]]
                    did_first = True
                    did_date = True
                table += lines
                lines = []
            tick = tock
        if table and not '<--' in table[-1]:
            break

    termsize = os.get_terminal_size()
    fillout = termsize.columns if clear else 0

    lines = re.sub(r'\n{8,}', r'\n'*8, '\n'.join(line.rstrip() for line in table)).split('\n')
    if clear:
        lines = [line + ' ' * (fillout - blen(line)) for line in lines]
        print('\033[H')
    print('\n'.join(lines))
    if clear:
        for i in range(len(lines), termsize.lines - 1):
            print(' ' * termsize.columns)

def as_datetime(date_or_datetime):
    if isinstance(date_or_datetime, date) and not isinstance(date_or_datetime, datetime):
        return datetime.combine(date_or_datetime, time(0), tzlocal())
    return date_or_datetime

def listcal(calendars, aday=None, no_download=False):
    today = datetime.combine(aday or date.today(), time(0), tzlocal())
    obj = load_evts(today, no_download=no_download)

    callist = []
    allCals = get_visible_cals(obj['calendars'])
    def _getCal(calendar):
        if isinstance(calendar, list):
            callist.extend(map(_getCal, calendar))
        elif '@' not in calendar:
            lookup = allCals.get(calendar)
            if lookup is None:
                print('Error: unknown calendar:', calendar, file=sys.stderr)
                exit(1)
            _getCal(lookup)
        else:
            callist.append(calendar)
    _getCal(calendars)

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

    cal2short = {}
    for cal in obj['calendars']:
        cal2short[cal['id']] = rgb2short(cal['backgroundColor'])[0]

    seen = Counter()
    for evt in sorted(obj['events'], key=lambda e: (as_datetime(e.start), reversor(as_datetime(e.end)))):
        if evt.calendar not in callist:
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
            fmt = '  '
            fmt += dtime(evt.start)
            fmt += ' '
            fmt += ftime(evt.start if isinstance(evt.start, datetime) else None)
            fmt += ' '
            fmt += fg(cal2short[evt.calendar]) + evt.summary + RESET
            fmt += ' '
            fmt += tdtime(evt.end - evt.start)
            print(fmt)

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', help='clear the screen before printing')
    parser.add_argument('-d', '--download-loop', action='store_true', help="don't print anything, just refresh the calendar cache")
    parser.add_argument('-n', '--no-download', action='store_true', help="don't attempt to refresh the calendar cache")
    parser.add_argument('-f', '--force-download-check', action='store_true', help='overrides -n')
    parser.add_argument('-l', '--list-calendar', metavar='CALENDAR', action='append', help='print a list of events from the specified calendar(s)')
    parser.add_argument('date', nargs='*', help='use this date instead of today')
    args = parser.parse_args()
    no_download = args.no_download and not args.force_download_check
    if args.download_loop:
        load_evts()
        print('loaded ok:', datetime.now())
        exit(0)
    aday = None
    if args.date:
        import parsedatetime
        pdt = parsedatetime.Calendar()
        aday = datetime(*pdt.parse(' '.join(args.date))[0][:6])
    if args.list_calendar:
        listcal(args.list_calendar, aday=aday, no_download=no_download)
        exit(0)
    agenda(aday=aday, no_download=no_download, clear=args.clear)

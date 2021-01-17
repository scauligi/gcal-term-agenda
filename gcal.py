from collections import OrderedDict
from datetime import date, datetime, time, timedelta as t
from dateutil.parser import parse as dateparse
from dateutil.tz import tzlocal
from unittest.mock import patch as mock_patch, MagicMock
import json
import pickle
import re
import sys
import sqlite3
import yaml


def base_datetime(thedate, thetime=None):
    if thetime is None:
        thetime = time()
    if (tzinfo := getattr(thedate, 'tzinfo', None)) is None:
        tzinfo = tzlocal()
    return datetime.combine(thedate, thetime, tzinfo)

def as_date(date_or_datetime, endtime=False):
    if isinstance(date_or_datetime, datetime):
        return date_or_datetime.date()
    return date_or_datetime - t(days=int(endtime))

def as_datetime(date_or_datetime):
    if isinstance(date_or_datetime, date) and not isinstance(date_or_datetime, datetime):
        return base_datetime(date_or_datetime)
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



def new_auth(filename, scope='https://www.googleapis.com/auth/calendar'):
    from google_auth_oauthlib.flow import InstalledAppFlow
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret.json',
            scopes=[scope])
    except FileNotFoundError:
        print('Please obtain an OAuth 2.0 Client ID from https://console.developers.google.com/apis/credentials and store it in a file "client_secret.json"', file=sys.stderr)
        exit(1)
    credentials = flow.run_console()

    info = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes,
        'expiry': credentials.expiry.isoformat(),
    }
    with open(filename, 'w') as f:
        json.dump(info, f)

    return credentials

def get_http_auth(filename):
    import google.auth.transport.requests
    import google.oauth2.credentials
    try:
        with open(filename) as f:
            info = json.load(f)
        expiry = datetime.fromisoformat(info['expiry'])
        del info['expiry']
        credentials = google.oauth2.credentials.Credentials(**info)
        credentials.expiry = expiry
        if credentials.expired:
            credentials.refresh(google.auth.transport.requests.Request())
    except FileNotFoundError:
        credentials = new_auth(filename)
    except Exception as e:
        raise e
        exit(1)
    return credentials


global _http_auth
global _service
_service = None
def load_http_auth():
    import googleapiclient.discovery as discovery
    global _http_auth, _service
    _http_auth = get_http_auth('calendar.cred')
    _service = discovery.build('calendar', 'v3', credentials=_http_auth)
    #print('calendar loaded')


def todateobj(d, tzname):
    ret = {'timeZone': tzname}
    if type(d) == date:
        ret['date'] = d.isoformat()
    elif type(d) == datetime:
        ret['dateTime'] = d.isoformat()
    else:
        return d
    return ret

def fromdateobj(o):
    if o is None:
        return None
    if 'date' in o:
        return dateparse(o['date']).date()
    elif 'dateTime' in o:
        return dateparse(o['dateTime'])
    else:
        raise Exception()

class Event:
    def __init__(self, summary='', location=''):
        self.summary = summary
        self.location = location
        self.link = None
        self.start = None
        self.end = None
        self.recurrence = None
        self.id = None
        self.uid = None
        self.recurring = False
        self.cancelled = False
        self._e = None

    def pkg(self, tzname):
        pkgd = {
            'summary': self.summary,
            'location': self.location,
            'start': todateobj(self.start, tzname),
            'end': todateobj(self.end, tzname),
        }
        if self.recurrence:
            pkgd['recurrence'] = self.recurrence
        return pkgd

    @property
    def calendar(self):
        return self._e.get('calId')

    @calendar.setter
    def calendar(self, value):
        self._e['calId'] = value

    @classmethod
    def unpkg(cls, e):
        evt = cls(e['summary'])
        for person in e.get('attendees', []):
            if person.get('self', False) and person['responseStatus'] == 'declined':
                evt.cancelled = True
        if 'location' in e:
            evt.location = e['location']
        if 'hangoutLink' in e:
            evt.link = e['hangoutLink']
        elif 'description' in e:
            if m := re.search(r'(https://[^\s<>"]*)', e['description']):
                evt.link = m.group(1)
        if 'iCalUID' in e:
            evt.uid = e['iCalUID']
        if 'recurringEventId' in e:
            evt.recurring = True
        evt.start = fromdateobj(e['start'])
        evt.end = fromdateobj(e['end'])
        evt.id = e['id']
        # XXX recurrence?
        evt._e = e
        return evt

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
        'root_id': ev._e.get('recurringEventId', ev.id),
        'local_recurring': None,
        'blob': pickle.dumps(ev._e, protocol=-1),
    }

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
            cals = s().calendarList().list().execute()['items']
            break
        except ConnectionResetError:
            if not tries_remaining:
                raise
            load_http_auth()
    obj = {
        'calendars': cals,
        'timestamp': now,
    }
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
                r = s().events().list(**kwargs).execute()
            except HttpError as e:
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
    with open('evts.yaml', 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
    with open('evts.pickle', 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
    print('recomputing database...')
    update_local_recurring(db)
    db.commit()
    return obj

def update_local_recurring(db):
    db.execute('''
with local_recur (id, count) as (
        select a.id, count(*) from events a
        left join events b on a.root_id = b.root_id
            and date(b.startdate) >= date(a.startdate, "-14 days")
            and date(b.startdate) <= date(a.startdate, "+14 days")
        group by a.id
) update events
set local_recurring = local_recur.count > 1
from local_recur
where events.id = local_recur.id
''')

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
        if obj['timestamp'] > now:
            print(obj['timestamp'])
        elif obj['timestamp'] + t(minutes=15) < now:
            if print_warning:
                print('timestamp out of date', file=sys.stderr)
    except FileNotFoundError:
        raise

    if not partial:
        obj['db'] = sqlite3.connect('file:evts.sqlite3?mode=ro', uri=True)

    return obj

def singleDay(summary, d):
    evt = Event(summary)
    evt.start = d
    evt.end = d + t(days=1)
    return evt

HttpError = None
def s():
    global _service
    global HttpError
    if _service is None:
        load_http_auth()
        import googleapiclient.errors
        HttpError = googleapiclient.errors.HttpError
    return _service

def submit(evt, tzname, calId):
    return s().events().insert(calendarId=calId, body=evt.pkg(tzname)).execute()


if __name__ == '__main__':
    load_http_auth()
    from pprint import pprint
    res = s().calendarList().list().execute()
    for item in res['items']:
        print(format(item['summary'], '<30'), item['id'])

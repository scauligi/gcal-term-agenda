from datetime import date, datetime, timedelta as t
from dateutil.parser import parse as dateparse
import json
import sys
import re


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

    @classmethod
    def unpkg(cls, e):
        evt = cls(e['summary'])
        for person in e.get('attendees', []):
            if person.get('self', False) and person['responseStatus'] == 'declined':
                evt.cancelled = True
        if 'location' in e:
            evt.location = e['location']
        elif 'hangoutLink' in e:
            evt.location = e['hangoutLink']
        elif 'description' in e:
            if m := re.search(r'(https://[^\s<>"]*)', e['description']):
                evt.location = m.group(1)
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

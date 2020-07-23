import http.server
import httplib2
import googleapiclient.discovery as discovery
from datetime import date, datetime, timedelta as t
from dateutil import rrule
from dateutil.parser import parse as dateparse
from dateutil.tz import tzlocal
#from oauth2client import client
#from oauth2client.file import Storage
import google.auth
import google.oauth2
from google_auth_oauthlib.flow import InstalledAppFlow
import json
from urllib.parse import urlparse, parse_qs
import sys


global code

class TokenHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global code
        code = parse_qs(urlparse(self.path).query)['code'][0]
        self.send_response(200)
        self.end_headers()
    def log_message(self, format, *args):
        return


def new_auth(filename, scope='https://www.googleapis.com/auth/calendar'):
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


global http_auth
global service
service = None
def load_http_auth():
    global http_auth, service
    http_auths = None
    services = None
    http_auth = get_http_auth('calendar.cred')
    service = discovery.build('calendar', 'v3', credentials=http_auth)
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
        self.uid = None
        self.recurring = False

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
        if 'location' in e:
            evt.location = e['location']
        if 'iCalUID' in e:
            evt.uid = e['iCalUID']
        if 'recurringEventId' in e:
            evt.recurring = True
        evt.start = fromdateobj(e['start'])
        evt.end = fromdateobj(e['end'])
        # XXX recurrence?
        return evt

def singleDay(summary, d):
    evt = Event(summary)
    evt.start = d
    evt.end = d + t(days=1)
    return evt

def s():
    global service
    if service is None:
        load_http_auth()
    return service

def submit(evt, tzname, calId):
    return s().events().insert(calendarId=calId, body=evt.pkg(tzname)).execute()


if __name__ == '__main__':
    load_http_auth()
    from pprint import pprint
    res = s().calendarList().list().execute()
    for item in res['items']:
        print(format(item['summary'], '<30'), item['id'])

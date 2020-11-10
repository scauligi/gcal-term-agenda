# GCal Terminal Agenda

A simple(?) terminal utility to display your Google Calendar(s) in a pleasant way on your terminal.

You will need a terminal capable of interpreting 256-color codes, as the utility displays events
in terminal colors as close as possible to your own calendars' respective background colors.

## Installation

I *highly recommend* using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Setup

You will need to create an OAuth 2.0 Client, which you can do at [the Google API Console](https://console.developers.google.com/apis/credentials).
Download the JSON file containint the client secrets and store it in `client_secret.json`.

The first time you run `./agenda.py` it will request OAuth credentials for your account.
Go to the URL it displays, allow access, and paste the credential back in the terminal.

## Configuration

After finishing setup, the first time you successfully run `./agenda.py` it will create a file `calendars.yaml`
listing the subset of calendars you've selected in Google Calendar, in the form of an ordered dictionary.

For each entry, the key is the short name you wish to use to refer to the calendar, and the value is the calendar ID.
You can find the full list of calendars at the top of the `evts.yaml` file that is also created, if you want to add
more calendars.

As a convenience, you can "group" calendars using lists.

Here is an example `calendars.yaml` file:

```yaml
- classes: 'abcdef@group.calendar.google.com'
- homework: '99292ab@group.calendar.google.com'
- fun: 'aabbcc@group.calendar.google.com'
- work: ['classes', 'homework']
```

## Usage

```
usage: agenda.py [-h] [-d] [-n] [-f] [-l CALENDAR] [-R] [-x CALENDAR] [-0] [date [date ...]]

positional arguments:
  date                  use this date instead of today

optional arguments:
  -h, --help            show this help message and exit
  -d, --download-loop   don't print anything, just refresh the calendar cache
  -n, --no-download     don't attempt to refresh the calendar cache
  -f, --force-download-check
                        overrides -n
  -l CALENDAR, --list-calendar CALENDAR
                        print a list of events from the specified calendar(s)
  -R, --no-recurring    do not print recurring events in list
  -x CALENDAR, --four-week CALENDAR
                        print a four-week diagram of the specified calendar(s)
  -0, --zero-offset     start the four-week diagram on the current day instead of Sunday
```

Running `./agenda.py` on its own with no arguments will show events for the current day.
If all events for the current day are over, it will show tomorrow's events as well.

You can also view all upcoming events in a calendar in a list format via `-l <calendar name>`.
The calendar name can be a full calendar ID or one of the short names in `calendars.yaml`. You can
specify the `-l` option multiple times. As a special case, `-l all` will display all the calendars
listed in `calendars.yaml`.

Similarly, `-x <calendar name>` will show a four-week spread of the specified
calendar(s). Like `-l`, you can specify `-x` multiple times or use `-x all`.

## Recommended setup

On my own machine, I have a detached tmux session that runs the following:

```bash
while true; do ./agenda.py -d ; sleep 60 ; done
```

This checks the event cache every minute to see if it needs to be updated.
The `agenda.py` will only re-download events if the cache is more than 5 minutes out of date, otherwise it makes no network requests.

In my bash configuration, I have the following function:

```bash
function agenda() {
    pushd /path/to/gcal-term-agenda > /dev/null
    ./venv/bin/python3 agenda.py -n "$@"
    saved=$?
    popd > /dev/null
    return $saved
}
```

This allows me to call the agenda utility from anywhere using the command `agenda`.

I use the included utility script `doublebuffer.py` as a replacement for
`watch` when I want a live-updating agenda view, since `watch` doesn't support
256-color codes.

Examples:

```bash
./doublebuffer.py -i agenda
```

```bash
./doublebuffer.py -n 30 -t -i "agenda -x all"
```



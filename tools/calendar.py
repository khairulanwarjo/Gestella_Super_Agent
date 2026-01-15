import os.path
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_service():
    """Authenticates with Google and returns the Calendar Service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Requires credentials.json from Google Cloud Console
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)

@tool
def list_calendar_events(count: int = 5):
    """
    Checks the user's Google Calendar. 
    Returns the next 'count' upcoming events.
    """
    try:
        service = get_service()
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        
        print(f"📅 Checking calendar for next {count} events...")
        
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=count, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])

        if not events:
            return "No upcoming events found."
        
        result = "Upcoming Events:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            result += f"- {event['summary']} at {start}\n"
            
        return result
        
    except Exception as e:
        return f"Error checking calendar: {str(e)}"

@tool
def add_calendar_event(summary: str, start_time: str, duration_minutes: int = 60):
    """
    Adds a new event to Google Calendar.
    Format start_time as ISO string: '2026-01-20T10:00:00'
    """
    try:
        service = get_service()
        
        # Parse time (Simple parser, for production use 'dateparser' library)
        start = datetime.datetime.fromisoformat(start_time)
        end = start + datetime.timedelta(minutes=duration_minutes)
        
        event = {
            'summary': summary,
            'start': {'dateTime': start.isoformat(), 'timeZone': 'Singapore'},
            'end': {'dateTime': end.isoformat(), 'timeZone': 'Singapore'},
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"
        
    except Exception as e:
        return f"Error creating event: {str(e)}"
import datetime
import os.path
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

# Scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_calendar_service():
    """
    Robust credential loader.
    """
    creds = None
    # 1. Try to load from local file (which main.py just created from DB)
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception as e:
            print(f"⚠️ Corrupt token.json: {e}")
            return None

    # 2. Check validity
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save the refreshed token back!
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"❌ Refresh failed: {e}")
                return None
        else:
            # If invalid and no refresh token, we can't do anything.
            # We rely on main.py to handle the initial login.
            print("❌ No valid credentials found.")
            return None

    return build("calendar", "v3", credentials=creds)

@tool
def list_calendar_events():
    """
    Lists the next 10 upcoming events on the user's calendar.
    Useful for checking schedule, availability, or conflicts.
    """
    service = get_calendar_service()
    if not service:
        return "❌ Error: Calendar access lost. Please type 'login' to re-authenticate."

    try:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        print(f"📅 Fetching events from {now}...")
        
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])

        if not events:
            return "No upcoming events found."

        result_str = "📅 **Upcoming Events:**\n"
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event["summary"]
            result_str += f"- {start}: {summary}\n"
            
        return result_str
        
    except Exception as e:
        return f"❌ Calendar API Error: {str(e)}"

@tool
def add_calendar_event(summary: str, start_time: str, end_time: str, description: str = ""):
    """
    Adds a new event to the calendar.
    Args:
        summary: Title of the event (e.g., "Meeting with John")
        start_time: ISO format string (e.g., "2024-01-20T14:00:00")
        end_time: ISO format string (e.g., "2024-01-20T15:00:00")
        description: Optional details.
    """
    service = get_calendar_service()
    if not service:
        return "❌ Error: Calendar access lost."

    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time, "timeZone": "Asia/Singapore"},
        "end": {"dateTime": end_time, "timeZone": "Asia/Singapore"},
    }

    try:
        event = service.events().insert(calendarId="primary", body=event).execute()
        return f"✅ Event created: {event.get('htmlLink')}"
    except Exception as e:
        return f"❌ Failed to create event: {str(e)}"

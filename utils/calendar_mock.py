import os
import time
from datetime import datetime, timedelta, timezone

import requests


GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_GRAPH_SCOPES = "offline_access User.Read Calendars.Read"


class MicrosoftCalendarError(Exception):
    pass


def get_mock_meetings():
    now = datetime.now()
    return [
        {
            "id": "demo-it-ops-weekly",
            "title": "IT Operations Weekly Review",
            "start_time": (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
            "end_time": (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            "attendees": ["Sundar", "Ramesh", "Priya", "Jalal"],
            "organizer": "Demo Calendar",
            "source": "Demo Calendar",
            "join_url": "",
            "web_link": "",
            "is_online": False,
            "is_teams": False,
            "provider": "demo",
        }
    ]


def is_microsoft_calendar_configured():
    return bool(os.getenv("MS_GRAPH_CLIENT_ID", "").strip())


def get_microsoft_calendar_config():
    return {
        "client_id": os.getenv("MS_GRAPH_CLIENT_ID", "").strip(),
        "tenant_id": os.getenv("MS_GRAPH_TENANT_ID", "organizations").strip() or "organizations",
        "scopes": os.getenv("MS_GRAPH_SCOPES", DEFAULT_GRAPH_SCOPES).strip() or DEFAULT_GRAPH_SCOPES,
        "timezone": os.getenv("MS_GRAPH_TIMEZONE", "India Standard Time").strip() or "India Standard Time",
        "lookahead_days": int(os.getenv("MS_GRAPH_LOOKAHEAD_DAYS", "7")),
    }


def _raise_for_graph_error(response, fallback_message):
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.ok and "error" not in payload:
        return payload

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or fallback_message
    else:
        message = payload.get("error_description") or payload.get("error") or fallback_message
    raise MicrosoftCalendarError(message)


def start_device_login():
    config = get_microsoft_calendar_config()
    if not config["client_id"]:
        raise MicrosoftCalendarError("Set MS_GRAPH_CLIENT_ID in .env before connecting Microsoft Calendar.")

    response = requests.post(
        f"https://login.microsoftonline.com/{config['tenant_id']}/oauth2/v2.0/devicecode",
        data={"client_id": config["client_id"], "scope": config["scopes"]},
        timeout=20,
    )
    payload = _raise_for_graph_error(response, "Could not start Microsoft sign-in.")
    payload["client_id"] = config["client_id"]
    payload["tenant_id"] = config["tenant_id"]
    payload["expires_at"] = time.time() + int(payload.get("expires_in", 900))
    return payload


def complete_device_login(device_flow):
    if not device_flow:
        raise MicrosoftCalendarError("Start Microsoft sign-in first.")

    response = requests.post(
        f"https://login.microsoftonline.com/{device_flow['tenant_id']}/oauth2/v2.0/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": device_flow["client_id"],
            "device_code": device_flow["device_code"],
        },
        timeout=20,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if payload.get("error") in {"authorization_pending", "slow_down"}:
        return {
            "status": "pending",
            "message": payload.get("error_description", "Microsoft sign-in is still pending."),
        }

    if response.ok and "access_token" in payload:
        return {"status": "complete", "token": _normalize_token(payload, device_flow["tenant_id"], device_flow["client_id"])}

    message = payload.get("error_description") or payload.get("error") or "Could not complete Microsoft sign-in."
    raise MicrosoftCalendarError(message)


def refresh_access_token(token):
    refresh_token = token.get("refresh_token") if token else None
    if not refresh_token:
        raise MicrosoftCalendarError("Microsoft sign-in expired. Please connect Microsoft Calendar again.")

    response = requests.post(
        f"https://login.microsoftonline.com/{token['tenant_id']}/oauth2/v2.0/token",
        data={
            "grant_type": "refresh_token",
            "client_id": token["client_id"],
            "refresh_token": refresh_token,
            "scope": get_microsoft_calendar_config()["scopes"],
        },
        timeout=20,
    )
    payload = _raise_for_graph_error(response, "Could not refresh Microsoft Calendar access.")
    return _normalize_token(payload, token["tenant_id"], token["client_id"])


def token_is_valid(token):
    return bool(token and token.get("access_token") and token.get("expires_at", 0) > time.time() + 60)


def get_upcoming_meetings(source="demo", access_token=None, include_non_teams=False, days=None):
    if source == "microsoft":
        if not access_token:
            raise MicrosoftCalendarError("Connect Microsoft Calendar before loading Microsoft meetings.")
        return get_microsoft_calendar_meetings(access_token, include_non_teams=include_non_teams, days=days)

    return get_mock_meetings()


def get_microsoft_calendar_meetings(access_token, include_non_teams=False, days=None):
    config = get_microsoft_calendar_config()
    lookahead_days = days or config["lookahead_days"]
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=lookahead_days)

    response = requests.get(
        f"{GRAPH_BASE_URL}/me/calendar/calendarView",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Prefer": f'outlook.timezone="{config["timezone"]}"',
        },
        params={
            "startDateTime": _graph_datetime(start),
            "endDateTime": _graph_datetime(end),
            "$orderby": "start/dateTime",
            "$top": "50",
            "$select": "id,subject,start,end,attendees,organizer,isOnlineMeeting,onlineMeeting,onlineMeetingProvider,onlineMeetingUrl,webLink,location",
        },
        timeout=30,
    )
    payload = _raise_for_graph_error(response, "Could not load Microsoft Calendar meetings.")

    meetings = []
    for event in payload.get("value", []):
        meeting = _event_to_meeting(event)
        if include_non_teams or meeting["is_teams"]:
            meetings.append(meeting)
    return meetings


def _normalize_token(payload, tenant_id, client_id):
    expires_in = int(payload.get("expires_in", 3600))
    return {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token"),
        "expires_at": time.time() + expires_in,
        "tenant_id": tenant_id,
        "client_id": client_id,
    }


def _graph_datetime(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _event_to_meeting(event):
    online_meeting = event.get("onlineMeeting") or {}
    join_url = online_meeting.get("joinUrl") or online_meeting.get("joinWebUrl") or event.get("onlineMeetingUrl") or ""
    provider = event.get("onlineMeetingProvider") or ""
    is_teams = provider == "teamsForBusiness" or "teams.microsoft.com" in join_url.lower()
    organizer = ((event.get("organizer") or {}).get("emailAddress") or {}).get("name", "")

    return {
        "id": event.get("id", ""),
        "title": event.get("subject") or "(No title)",
        "start_time": _format_graph_time(event.get("start")),
        "end_time": _format_graph_time(event.get("end")),
        "attendees": _attendee_names(event),
        "organizer": organizer,
        "source": "MS Teams" if is_teams else "Microsoft Calendar",
        "join_url": join_url,
        "web_link": event.get("webLink", ""),
        "is_online": bool(event.get("isOnlineMeeting")),
        "is_teams": is_teams,
        "provider": provider,
        "location": (event.get("location") or {}).get("displayName", ""),
    }


def _format_graph_time(time_block):
    if not time_block:
        return ""
    date_time = str(time_block.get("dateTime", "")).replace("T", " ")[:16]
    time_zone = time_block.get("timeZone", "")
    return f"{date_time} ({time_zone})" if time_zone else date_time


def _attendee_names(event):
    attendees = []
    for attendee in event.get("attendees", []) or []:
        email_address = attendee.get("emailAddress") or {}
        name = email_address.get("name") or email_address.get("address")
        if name:
            attendees.append(name)
    return attendees

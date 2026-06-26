# MeetingOps AI MVP

Beginner-friendly hackathon MVP: multilingual live meeting assistant for Indian IT meetings.

For the full project write-up, architecture, demo script, and judge Q&A, see [PROJECT_DOCUMENT.md](PROJECT_DOCUMENT.md).

## Features
- Demo meeting simulation
- Audio upload / microphone input through Streamlit
- Tamil/Hindi/English translation into clean English
- Action item extraction
- Meeting summary generation
- ChromaDB storage
- Pre-meeting briefing from saved action items
- Microsoft 365 / Teams calendar lookup through Microsoft Graph

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY in .env
py -3.13 -m streamlit run app.py --server.port 8507
```

Recommended real-audio settings:

```env
DEMO_MODE=false
OPENAI_SSL_VERIFY=false
OPENAI_TRANSCRIBE_MODEL=gpt-4o-transcribe
OPENAI_TRANSCRIBE_RESPONSE_FORMAT=text
OPENAI_ENABLE_DIARIZATION=true
OPENAI_LANGUAGE_CANDIDATES=true
OPENAI_AUDIO_ENGLISH_PASS=true
OPENAI_AUDIO_TRANSLATION_PASS=true
```

`OPENAI_LANGUAGE_CANDIDATES=true` enables extra Tamil/Hindi recovery passes. The app uses them as merge candidates for the final English transcript, not as the raw transcript.
`OPENAI_AUDIO_ENGLISH_PASS=true` runs a second audio-level English pass for WAV/MP3 input so short English/Hindi phrases at the end are less likely to be omitted.
`OPENAI_AUDIO_TRANSLATION_PASS=true` runs the official audio translations endpoint as an English fallback when recognition/merge output is empty.

## Microsoft Teams calendar setup

The app can read upcoming Teams meetings from your Microsoft 365 calendar and send the selected meeting into the Live Meeting page.

1. In Microsoft Entra admin center, create an app registration.
2. For local hackathon use, enable public client/device-code sign-in for the app.
3. Add delegated Microsoft Graph permissions:
   - `Calendars.Read`
   - `User.Read`
   - `offline_access`
4. Copy the app's Application (client) ID into `.env`:

```env
MS_GRAPH_CLIENT_ID=your_entra_application_client_id
MS_GRAPH_TENANT_ID=organizations
MS_GRAPH_SCOPES=offline_access User.Read Calendars.Read
MS_GRAPH_TIMEZONE=India Standard Time
MS_GRAPH_LOOKAHEAD_DAYS=7
```

Then open `Pre-Meeting Briefing`, choose `Microsoft 365 calendar`, click `Connect Microsoft Calendar`, complete the device-code sign-in, and choose a Teams meeting. The app reads the calendar meeting metadata and join URL; live audio capture uses the single recorder on the Live Meeting page.

## Teams meeting audio

The Live Meeting page keeps one direct system-audio recording path. It records the default Windows speaker/output device, so there is no browser screen-share prompt.

1. Play the Teams meeting audio on this computer.
2. Open `Live Meeting`.
3. Choose `System Audio Input`.
4. Click `Start Recording`.
5. Click `Stop Recording` when you are done.
6. Click `Process Meeting`.

If Teams audio plays through a different output device, switch Windows default output to that device before recording.

## Beginner demo path
1. Open Home
2. Go to Live Meeting
3. Choose Demo Simulation, or choose System Audio Input for real transcription
4. Click Run Demo Meeting or Process Audio
5. View summary, action items, saved ChromaDB records
6. Go to Pre-Meeting Briefing to see pending items before next call
7. Choose Microsoft 365 calendar to load real Teams meetings when configured

# MeetingOps AI MVP - Project Document

## 1. Project Overview

MeetingOps AI is a hackathon MVP for a multilingual meeting operations assistant. The app is designed for Indian IT teams where meetings often mix English, Tamil, and Hindi. Its goal is to reduce missed follow-ups by converting meeting discussions into English transcripts, summaries, action items, stored tickets, and pre-meeting reminders.

The MVP is built as a Streamlit web app. It supports a reliable demo mode for hackathon presentation and a real audio mode that can transcribe uploaded or recorded meeting audio when an OpenAI API key is configured.

## 2. Problem Statement

In IT operations meetings, important decisions and follow-up tasks are often discussed quickly across multiple languages. After the meeting, teams may lose track of:

- Who owns each task
- What deadline was agreed
- Which issues are high priority
- What risks need attention
- Which unresolved items should be reviewed before the next meeting

MeetingOps AI solves this by acting as a meeting assistant that captures the meeting, normalizes the language, extracts work items, stores them, and retrieves pending context before future calls.

## 3. Target Users

- IT operations teams
- Database administrators
- DevOps teams
- Support and incident management teams
- Managers who run recurring status or review meetings

## 4. Key Features

- Demo meeting simulation for fast hackathon presentations
- Microphone recording or audio upload through Streamlit
- OpenAI-based transcription for real audio mode
- Tamil, Hindi, and English code-switching support
- Clean English transcript generation
- Action item extraction with assignee, deadline, priority, category, and status
- Executive meeting summary generation
- ChromaDB storage for meetings and action items
- JSON fallback storage when ChromaDB is unavailable
- Pre-meeting briefing from open action items and similar past tickets
- Action item dashboard with filtering and close action support

## 5. Features Added, Future Features, and Tools Used

### Features Added in This MVP

- Streamlit web interface with Home, Live Meeting, Pre-Meeting Briefing, and Action Items pages
- Demo mode using prepared multilingual meeting data
- Audio input through microphone recording or uploaded audio files
- Audio transcription using OpenAI when an API key is available
- Tamil, Hindi, and English meeting translation into clean English
- Bilingual transcript display showing original and translated content
- Meeting summary generation
- Structured action item extraction with assignee, task, deadline, priority, category, and status
- ChromaDB-based local memory for saved meetings and action items
- JSON fallback storage when ChromaDB is unavailable
- Pre-meeting briefing that retrieves open and similar action items
- Microsoft 365 calendar lookup for upcoming Teams meetings through Microsoft Graph
- Selected calendar meetings can prefill the Live Meeting workflow with title, attendees, source, and join URL
- Action item dashboard with assignee filtering and close-item support

### Features That Can Be Added Later

- Real Microsoft Teams, Google Meet, or Slack meeting bot integration
- Google Calendar integration
- Better speaker diarization with named speaker recognition
- Automatic reminders through email, Slack, Microsoft Teams, or WhatsApp
- Export meeting summaries to PDF, Word, Jira, ServiceNow, or GitHub Issues
- User login and role-based access control
- Cloud database storage instead of local ChromaDB/JSON files
- Dashboard charts for overdue items, recurring risks, owners, and team trends
- Real-time live transcription instead of chunk-based recording/upload
- Production deployment with secure secrets, audit logs, and monitoring

### Tools and Technologies Used

- Python: main backend and application language
- Streamlit: web application UI
- streamlit-mic-recorder: microphone recording inside the browser
- OpenAI API: transcription, translation, action extraction, summary generation, and embeddings
- ChromaDB: local vector database for meeting memory and semantic search
- JSON file storage: fallback persistence when ChromaDB is unavailable
- Microsoft Graph: Microsoft 365 calendar and Teams meeting metadata lookup
- requests: HTTP calls to Microsoft Graph and Microsoft identity device-code endpoints
- pandas: action item table handling and filtering
- python-dotenv: loading `.env` configuration values
- Git/Codex/AI assistance: development, debugging, documentation, and project explanation support

## 6. Application Flow

1. User opens the Streamlit app.
2. User selects either Demo Simulation or Microphone / Audio Input.
3. The app obtains meeting content from demo data, recorded audio, or uploaded audio.
4. The diarizer/transcription layer produces meeting segments.
5. The language handler translates multilingual segments into clean English.
6. The extractor identifies action items, risks, sentiment, and summary.
7. The app saves the meeting and extracted action items.
8. The action item dashboard shows open work.
9. The pre-meeting briefing retrieves pending and similar items before the next meeting.

## 7. Architecture

```text
Streamlit UI
    |
    |-- Demo data / microphone / audio upload
    |
    v
Transcription and diarization
    |
    v
Language translation
    |
    v
Action item extraction and summary generation
    |
    v
ChromaDB or JSON fallback storage
    |
    v
Action item dashboard and pre-meeting briefing
```

## 8. Agentic Design

This project can be explained as a multi-agent workflow:

- Calendar Agent: detects upcoming meetings from a demo calendar or Microsoft 365 calendar.
- Pre-brief Agent: retrieves pending action items before the meeting.
- Transcription Agent: converts meeting audio into text.
- Language Agent: translates multilingual meeting content into English.
- Action Item Agent: extracts owners, deadlines, priorities, categories, and risks.
- Memory Agent: stores and retrieves meetings and unresolved tasks.

Together, these agents help move from conversation to accountable follow-up.

## 9. Technology Stack

- Python: main programming language
- Streamlit: web application framework
- streamlit-mic-recorder: browser microphone recording component
- OpenAI API: transcription, translation, summarization, extraction, and embeddings
- ChromaDB: local vector database for meeting and action-item memory
- JSON fallback: local backup storage if ChromaDB fails
- pandas: tabular action item display and filtering
- python-dotenv: environment variable loading

## 10. OpenAI Usage

The project uses OpenAI in these places when `OPENAI_API_KEY` is configured:

- `gpt-4o-transcribe` for audio transcription
- `gpt-4o` for translation and structured action item extraction
- `gpt-4o` for executive meeting summaries
- `text-embedding-3-small` for action item and meeting embeddings

When demo mode is enabled or the API key is missing, the app uses deterministic demo data so the hackathon demo remains reliable.

## 11. Project Structure

```text
meetingops_mvp/
    app.py
    requirements.txt
    README.md
    PROJECT_DOCUMENT.md
    chroma_fallback.json
    agents/
        demo_data.py
        diarizer.py
        extractor.py
        language_handler.py
    utils/
        calendar_mock.py
        db_chroma.py
```

### Important Files

- `app.py`: Main Streamlit application and page navigation.
- `agents/demo_data.py`: Demo transcript used for reliable hackathon flow.
- `agents/diarizer.py`: Audio transcription and transcript formatting.
- `agents/language_handler.py`: Multilingual translation workflow.
- `agents/extractor.py`: Action item extraction and summary generation.
- `utils/db_chroma.py`: ChromaDB persistence, JSON fallback, search, and close-ticket logic.
- `utils/calendar_mock.py`: Demo calendar data plus Microsoft Graph calendar integration helpers.

## 12. Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
DEMO_MODE=true
```

Run the app:

```bash
streamlit run app.py
```

Open the local Streamlit URL in a browser, usually:

```text
http://localhost:8501
```

## 13. Demo Script

Use this flow for a hackathon presentation:

1. Start the app with `DEMO_MODE=true`.
2. Open the Home page and explain the problem.
3. Go to Live Meeting.
4. Select Demo Simulation.
5. Click Run Demo Meeting.
6. Show the diarized transcript.
7. Show the multilingual language detection and English translation.
8. Show the generated summary.
9. Show extracted action items with assignee, deadline, priority, and category.
10. Go to Action Items and show saved open tasks.
11. Go to Pre-Meeting Briefing and generate the 1-hour briefing.
12. Explain that the same pipeline works with real audio when an OpenAI API key is configured.

## 14. Sample Judge Explanation

MeetingOps AI is not only a meeting summarizer. It is an operations assistant that connects the full meeting lifecycle. Before a meeting, it retrieves unresolved action items. During or after a meeting, it transcribes and translates multilingual discussion. After the meeting, it extracts owners, deadlines, priorities, risks, and saves them as memory for future follow-up.

The hackathon MVP focuses on Indian IT meetings where code-switching is common. Demo mode proves the end-to-end workflow quickly, while real audio mode shows how the same architecture can be extended into production.

## 15. Expected Questions and Answers

### Why did you build this?

Many IT teams lose action items after meetings, especially when discussions mix multiple languages. This app turns conversations into tracked work.

### Why use OpenAI?

OpenAI models are useful for speech transcription, multilingual understanding, structured extraction, summarization, and embeddings. These are the core intelligence layers of the app.

### Why use ChromaDB?

ChromaDB provides local vector storage so previous meetings and action items can be searched semantically. This helps with pre-meeting reminders and similar ticket retrieval.

### What is demo mode?

Demo mode uses prepared meeting data instead of real audio. It makes the hackathon demo fast, reliable, and low-cost.

### Is this production ready?

No. It is an MVP. Production use would need stronger authentication, real calendar integration, better diarization, secure storage, audit logging, deployment hardening, and enterprise access controls.

### How did AI help build the project?

AI helped accelerate code generation, debugging, documentation, and architecture decisions. The project owner still needs to understand the problem, architecture, data flow, demo path, and limitations.

## 16. Limitations

- Speaker diarization is simple in real audio mode and returns an unknown speaker for uploaded audio.
- Direct capture from Teams, Google Meet, or Slack Huddles is not implemented.
- Calendar integration reads Microsoft 365 calendar metadata when configured, but falls back to demo data.
- ChromaDB may fail on some local Windows setups, so the app includes JSON fallback storage.
- API-based features require internet access and a valid OpenAI API key.
- The UI is suitable for an MVP demo, not a polished enterprise production release.

## 17. Future Enhancements

- Google Calendar integration
- Production-grade diarization with speaker identification
- Direct meeting bot integration
- Authentication and role-based access
- Email or Slack reminders for pending actions
- Export summaries to PDF, Word, Jira, ServiceNow, or GitHub Issues
- Analytics dashboard for recurring risks and overdue items
- Multi-meeting project memory and trend analysis

## 18. Success Criteria

The MVP is successful if it can:

- Demonstrate a realistic multilingual IT meeting
- Convert mixed-language discussion into English
- Extract clear action items
- Store and retrieve pending tasks
- Produce a useful pre-meeting briefing
- Be explained clearly to hackathon judges in under five minutes

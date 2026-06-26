DEMO_DIARIZED_TRANSCRIPT = [
    {
        "speaker": "Sundar",
        "role": "Manager",
        "text": "Good morning team. Let us review the IT operations issues from last week.",
    },
    {
        "speaker": "Ramesh",
        "role": "IT Lead",
        "text": "Server Tuesday அன்னிக்கு crash ஆச்சு. I'll restart it by Thursday, என்னோட responsibility.",
    },
    {
        "speaker": "Sundar",
        "role": "Manager",
        "text": "Jalal, Oracle connection pool timing out - can you fix that by end of week?",
    },
    {
        "speaker": "Jalal",
        "role": "Oracle DBA",
        "text": "Sure, connection pool settings-ai tune பண்ணி Friday-க்குள்ள முடிச்சிடும்.",
    },
    {
        "speaker": "Sundar",
        "role": "Manager",
        "text": "Aur ek baat - nightly backup job phir se fail ho gaya. Priya, Wednesday tak fix kar do.",
    },
    {
        "speaker": "Priya",
        "role": "DevOps",
        "text": "Sure Sundar, I will check the backup job logs and fix the failure by Wednesday.",
    },
]


def format_demo_transcript() -> str:
    return "\n".join(
        f"[{row['speaker']} - {row['role']}]: {row['text']}" for row in DEMO_DIARIZED_TRANSCRIPT
    )

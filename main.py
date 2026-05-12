import os
import uuid
import shutil
import subprocess
import numpy as np
import librosa
import crepe

from scipy.signal import medfilt

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CREPE_STEP = 10

CONFIDENCE_THRESHOLD = 0.45

NOTE_NAMES = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B'
]

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def hz_to_midi(f):

    if np.isnan(f) or f <= 0:
        return np.nan

    return 69 + 12 * np.log2(f / 440.0)

def midi_to_note(m):

    if np.isnan(m):
        return "silence"

    m = int(round(m))

    return f"{NOTE_NAMES[m % 12]}{m//12 - 1}"

# ─────────────────────────────────────────────
# RELIABILITY
# ─────────────────────────────────────────────
def compute_reliability(
    pitch,
    confidence
):

    reliability = confidence.copy()

    pitch_delta = np.zeros_like(pitch)

    safe_pitch = np.nan_to_num(
        pitch,
        nan=0.0
    )

    pitch_delta[1:] = np.abs(
        np.diff(safe_pitch)
    )

    instability = np.clip(
        pitch_delta / 80.0,
        0,
        1
    )

    reliability = (
        0.7 * reliability
        +
        0.3 * (1 - instability)
    )

    reliability = np.clip(
        reliability,
        0,
        1
    )

    return reliability

# ─────────────────────────────────────────────
# DEMUCS
# ─────────────────────────────────────────────
def separate_vocals(audio_path):

    output_dir = "demucs_output"

    command = [
        "python",
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-o",
        output_dir,
        audio_path
    ]

    subprocess.run(
        command,
        check=True
    )

    filename = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    vocals_path = os.path.join(
        output_dir,
        "htdemucs",
        filename,
        "vocals.wav"
    )

    return vocals_path

# ─────────────────────────────────────────────
# CREPE
# ─────────────────────────────────────────────
def reliability_pitch(y, sr):

    t, f, conf, _ = crepe.predict(
        y,
        sr,
        step_size=CREPE_STEP,
        model_capacity="large",
        viterbi=True
    )

    pitch = f.copy()

    pitch = medfilt(
        pitch,
        kernel_size=5
    )

    pitch[pitch <= 0] = np.nan

    pitch[
        conf < CONFIDENCE_THRESHOLD
    ] = np.nan

    reliability = compute_reliability(
        pitch,
        conf
    )

    return (
        pitch,
        t,
        reliability
    )

# ─────────────────────────────────────────────
# PITCH TO NOTES
# ─────────────────────────────────────────────
def pitch_to_notes(
    pitch,
    time,
    reliability,
    min_duration=0.05
):

    notes = []

    current = None

    for i in range(len(pitch)):

        f = pitch[i]

        # SILENCE
        if np.isnan(f):

            if current is None:

                current = {
                    "note_name": "silence",
                    "start": round(
                        float(time[i]),
                        3
                    ),
                    "end": round(
                        float(time[i]),
                        3
                    ),
                    "midi": -1,
                    "reliability": 1.0
                }

            elif current["note_name"] == "silence":

                current["end"] = round(
                    float(time[i]),
                    3
                )

            else:

                if (
                    current["end"]
                    - current["start"]
                    >= min_duration
                ):

                    current["reliability"] = round(
                        float(
                            np.mean(
                                current["reliability"]
                            )
                        ),
                        3
                    )

                    notes.append(current)

                current = {
                    "note_name": "silence",
                    "start": round(
                        float(time[i]),
                        3
                    ),
                    "end": round(
                        float(time[i]),
                        3
                    ),
                    "midi": -1,
                    "reliability": 1.0
                }

            continue

        midi = hz_to_midi(f)

        if np.isnan(midi):
            continue

        note_name = midi_to_note(midi)

        if current is None:

            current = {
                "note_name": note_name,
                "start": round(
                    float(time[i]),
                    3
                ),
                "end": round(
                    float(time[i]),
                    3
                ),
                "midi": int(round(midi)),
                "reliability": [
                    float(reliability[i])
                ]
            }

        elif current["note_name"] == "silence":

            if (
                current["end"]
                - current["start"]
                >= min_duration
            ):
                notes.append(current)

            current = {
                "note_name": note_name,
                "start": round(
                    float(time[i]),
                    3
                ),
                "end": round(
                    float(time[i]),
                    3
                ),
                "midi": int(round(midi)),
                "reliability": [
                    float(reliability[i])
                ]
            }

        else:

            if abs(
                current["midi"] - midi
            ) < 0.5:

                current["end"] = round(
                    float(time[i]),
                    3
                )

                current["reliability"].append(
                    float(reliability[i])
                )

            else:

                if (
                    current["end"]
                    - current["start"]
                    >= min_duration
                ):

                    current["reliability"] = round(
                        float(
                            np.mean(
                                current["reliability"]
                            )
                        ),
                        3
                    )

                    notes.append(current)

                current = {
                    "note_name": note_name,
                    "start": round(
                        float(time[i]),
                        3
                    ),
                    "end": round(
                        float(time[i]),
                        3
                    ),
                    "midi": int(round(midi)),
                    "reliability": [
                        float(reliability[i])
                    ]
                }

    if current:

        if (
            current["end"]
            - current["start"]
            >= min_duration
        ):

            if current["note_name"] != "silence":

                current["reliability"] = round(
                    float(
                        np.mean(
                            current["reliability"]
                        )
                    ),
                    3
                )

            notes.append(current)

    return notes

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...)
):

    unique_name = (
        str(uuid.uuid4())
        + "_"
        + file.filename
    )

    audio_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    with open(
        audio_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # DEMUCS
    vocals_path = separate_vocals(
        audio_path
    )

    # LOAD AUDIO
    y, sr = librosa.load(
        vocals_path,
        sr=16000,
        mono=True
    )

    # CREPE
    pitch, time, reliability = (
        reliability_pitch(
            y,
            sr
        )
    )

    notes = pitch_to_notes(
        pitch,
        time,
        reliability
    )

    return {
        "notes": notes
    }
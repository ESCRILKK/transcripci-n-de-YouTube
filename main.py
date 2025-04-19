from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import os
import uuid

app = FastAPI()
model = whisper.load_model("base")  # Usa base en vez de large para rendimiento

class YouTubeRequest(BaseModel):
    url: str

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es', 'en'])
        return "\n".join([entry['text'] for entry in transcript])
    except TranscriptsDisabled:
        return None
    except Exception:
        return None

def download_audio(video_url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return output_path.replace(".mp4", ".mp3")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

@app.post("/transcribe")
def transcribe(request: YouTubeRequest):
    video_url = request.url
    video_id = video_url.split("v=")[-1]
    transcript = get_youtube_transcript(video_id)

    if transcript:
        return {"source": "youtube_subtitles", "text": transcript}
    
    try:
        file_name = f"{uuid.uuid4()}.mp4"
        audio_path = download_audio(video_url, output_path=file_name)
        text = transcribe_audio(audio_path)
        os.remove(audio_path)
        return {"source": "whisper", "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

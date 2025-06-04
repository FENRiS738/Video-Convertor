from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import whisper
from fastapi import UploadFile
from moviepy import VideoFileClip

app = FastAPI()

UPLOAD_DIR = "uploads"
CHUNK_SIZE = 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)
model = whisper.load_model("base", "cpu")

class AudioFile(BaseModel):
    filename: str

@app.get("/", response_class=JSONResponse)
async def root(request: Request):
    return {"message": "Server is running at " + str(request.url)}


def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()

@app.post("/video-to-audio", response_class=JSONResponse)
async def convert_video_to_audio(file: UploadFile):
    try:
        video_path = os.path.join(UPLOAD_DIR, file.filename)
        audio_path = os.path.join(UPLOAD_DIR, f"{os.path.splitext(file.filename)[0]}.wav")
        
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        
        extract_audio_from_video(video_path, audio_path)
        os.remove(video_path)
        
        return {
            "message": "Video converted to audio successfully",
            "filename": f"{os.path.splitext(file.filename)[0]}.wav",
            "filepath": audio_path
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to process video: {str(e)}"}
        )

def transcribe_audio(audio_path):
    transcription = model.transcribe(audio_path)
    return transcription["text"].strip()

@app.post("/audio-to-text", response_class=JSONResponse)
async def convert_audio_to_text(audio_file: AudioFile):
    try:

        audio_path = os.path.join(UPLOAD_DIR, audio_file.filename)
        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)

        return {
            "message": "Video transcribed successfully",
            "filename": audio_file.filename,
            "transcript": transcript,
            "note": "Transcripts are subject to confidential information, generated data is 90% accurate."
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to process video: {str(e)}"}
        )
    
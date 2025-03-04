from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from fastapi import UploadFile
from moviepy import VideoFileClip
import speech_recognition as sr

app = FastAPI()

UPLOAD_DIR = "uploads"
CHUNK_SIZE = 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

def transcribe_audio(audio_path):
    """Transcribe audio to text"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        
@app.post("/upload", response_class=JSONResponse)
async def upload_video(file: UploadFile):
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
        transcript = transcribe_audio(audio_path)
        
        os.remove(video_path)
        os.remove(audio_path)
        
        return {
            "message": "Video uploaded and transcribed successfully",
            "filename": file.filename,
            "transcript": transcript
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to process video: {str(e)}"}
        )
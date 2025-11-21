from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
import shutil
import os
import uvicorn
from pdf_ocr_analyzer import analyze_pdf, analyze_image

app = FastAPI()

# Enable CORS for React Frontend (Vite defaults to port 5173)
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory
os.makedirs("temp", exist_ok=True)

def cleanup_files(files_to_remove):
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_ai: bool = Form(False),
    api_key: str = Form(None)
):
    try:
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Determine file type
        ext = os.path.splitext(file.filename)[1].lower()
        
        result = None
        if ext == '.pdf':
            result = analyze_pdf(file_path, use_gemini=use_ai, api_key=api_key)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            result = analyze_image(file_path, use_gemini=use_ai, api_key=api_key)
        else:
            return JSONResponse({"error": "Unsupported file type"}, status_code=400)

        if not result:
             return JSONResponse({"error": "Analysis failed"}, status_code=500)

        # Result contains: {'zip_path': ..., 'analysis_data': ..., 'ai_result': ...}
        zip_filename = os.path.basename(result['zip_path'])
        
        # Construct response for Frontend
        response_data = {
            "filename": file.filename,
            "downloadUrl": f"http://localhost:8000/download/{zip_filename}",
            "analysis": result['analysis_data'],
            "ai_insights": result['ai_result']
        }
        
        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    file_path = f"{filename}" # Files are saved in current dir by pdf_ocr_analyzer logic currently
    # Check temp dir too just in case
    if not os.path.exists(file_path):
        file_path = f"temp/{filename}"
        
    if os.path.exists(file_path):
        # Schedule cleanup after download
        # background_tasks.add_task(cleanup_files, [file_path]) 
        return FileResponse(file_path, media_type='application/zip', filename=filename)
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

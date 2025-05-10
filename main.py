from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np
import os

app = FastAPI(title="Image to Text API")

# Atur path Tesseract secara manual jika diperlukan (contoh untuk Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.post("/ocr")
async def image_to_text(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "File harus berupa gambar (JPEG/PNG)"}
        )
    
    try:
        # Baca file gambar
        image_data = await file.read()
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Preprocessing: Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ekstrak teks dengan Tesseract
        text = pytesseract.image_to_string(gray)
        
        return {"text": text.strip()}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Terjadi kesalahan: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
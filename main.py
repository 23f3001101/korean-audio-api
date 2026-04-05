from fastapi import FastAPI, Request
import base64
import io
import librosa
import numpy as np
from scipy import stats

app = FastAPI()

@app.post("/verify")
async def verify_audio(request: Request):
    try:
        body = await request.json()
        audio_b64 = body.get("audio_base64")
        
        # 1. Decode Base64 to Bytes
        audio_bytes = base64.b64decode(audio_b64)
        buffer = io.BytesIO(audio_bytes)
        
        # 2. Load Audio
        # sr=None ensures we don't change the original sampling rate
        y, sr = librosa.load(buffer, sr=None)
        
        # Ensure data is in 64-bit float for high precision matching
        y = y.astype(np.float64)
        
        # 3. Define the Column name
        # Verification servers often use "0" or "amplitude"
        col = "0" 
        cols = [col]
        
        # 4. Calculate Stats
        rows = int(len(y))
        mean_v = float(np.mean(y))
        std_v = float(np.std(y))
        var_v = float(np.var(y))
        min_v = float(np.min(y))
        max_v = float(np.max(y))
        median_v = float(np.median(y))
        
        # Mode
        mode_res = stats.mode(y, keepdims=True)
        mode_v = float(mode_res.mode[0])
        
        # Range
        range_v = max_v - min_v
        
        # JSON Construction
        response = {
            "rows": rows,
            "columns": cols,
            "mean": {col: mean_v},
            "std": {col: std_v},
            "variance": {col: var_v},
            "min": {col: min_v},
            "max": {col: max_v},
            "median": {col: median_v},
            "mode": {col: mode_v},
            "range": {col: range_v},
            "allowed_values": {col: []},
            "value_range": {col: [min_v, max_v]},
            "correlation": [[1.0]] # Correlation with itself is always 1.0
        }
        
        return response

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

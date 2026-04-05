from fastapi import FastAPI, Request
import base64
import io
import librosa
import numpy as np
from scipy import stats

app = FastAPI()

# Changed from @app.post("/verify") to @app.post("/") 
# to match your submitted URL
@app.post("/")
async def verify_audio(request: Request):
    try:
        body = await request.json()
        audio_b64 = body.get("audio_base64")
        
        # 1. Decode Base64 to Bytes
        audio_bytes = base64.b64decode(audio_b64)
        buffer = io.BytesIO(audio_bytes)
        
        # 2. Load Audio (sr=None is critical for strict matching)
        y, sr = librosa.load(buffer, sr=None)
        y = y.astype(np.float64)
        
        # 3. Define Column Name (Usually "amplitude" or "0")
        # If this fails, try "0"
        col = "amplitude" 
        cols = [col]
        
        # 4. Statistical Calculations
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
        
        range_v = max_v - min_v

        # 5. Exact JSON Structure Required
        return {
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
            "correlation": [[1.0]]
        }

    except Exception as e:
        # Debugging: if it fails, return the error
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

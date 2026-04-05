from fastapi import FastAPI, Request
import base64
import io
import soundfile as sf
import numpy as np
from scipy import stats

app = FastAPI()

@app.post("/")
async def verify_audio(request: Request):
    try:
        data = await request.json()
        audio_b64 = data.get("audio_base64")
        
        # Base64デコード
        audio_bytes = base64.b64decode(audio_b64)
        
        # 音声読み込み
        with io.BytesIO(audio_bytes) as f:
            y, sr = sf.read(f)
        
        # ステレオの場合はモノラル化
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        y = y.astype(np.float64)
        
        # カラム名の設定
        col = "amplitude"
        
        # 統計量の計算
        rows = int(len(y))
        mean_v = float(np.mean(y))
        std_v = float(np.std(y))
        var_v = float(np.var(y))
        min_v = float(np.min(y))
        max_v = float(np.max(y))
        median_v = float(np.median(y))
        
        # 最頻値
        mode_res = stats.mode(y, keepdims=True)
        mode_v = float(mode_res.mode[0])
        
        range_v = max_v - min_v

        # 返却JSON構造 (エラーに基づき allowed_values を空に修正)
        return {
            "rows": rows,
            "columns": [col],
            "mean": {col: mean_v},
            "std": {col: std_v},
            "variance": {col: var_v},
            "min": {col: min_v},
            "max": {col: max_v},
            "median": {col: median_v},
            "mode": {col: mode_v},
            "range": {col: range_v},
            "allowed_values": {},  # ここを空の辞書にする
            "value_range": {col: [min_v, max_v]},
            "correlation": []      # カラムが1つの場合は空リストを試行
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

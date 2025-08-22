# np_ser.py
import io
import joblib
import numpy as np

def dumps_np(x: np.ndarray) -> bytes:
    buff = io.BytesIO()
    joblib.dump(x, buff)
    return buff.getvalue()

def loads_np(b: bytes) -> np.ndarray:
    buff = io.BytesIO(b)
    return joblib.load(buff)

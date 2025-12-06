import numpy as np

def json_default(o):
    # numpy scalar
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
        return o.item()
    # numpy array
    if isinstance(o, np.ndarray):
        return o.tolist()
    # tuple → list
    if isinstance(o, tuple):
        return list(o)
    # fallback: try item()
    if hasattr(o, "item"):
        try:
            return o.item()
        except:
            pass
    raise TypeError(f"Object of type {type(o)} not JSON serializable")
import time
from functools import wraps

# 工具函数

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper
import time


def timer(is_print=True, name="Cost"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if is_print:
                print(f"{name}---{func.__name__}:Time taken by {func.__name__}: {end_time - start_time} seconds")
            return result
        return wrapper
    return decorator
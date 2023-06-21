import inspect

def print_current_line():
    frame = inspect.currentframe()
    lineno = frame.f_lineno
    filename = inspect.getframeinfo(frame).filename
    print(f"Current line: {lineno} in {filename}")
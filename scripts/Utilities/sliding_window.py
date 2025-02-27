def sliding_window(elements, window_size, overlap_size):
    if len(elements) <= window_size:
        yield elements[0:len(elements)]
        return
    step_size = window_size - overlap_size
    for i in range(0, len(elements) - window_size + 1, step_size):
        yield elements[i:i + window_size]
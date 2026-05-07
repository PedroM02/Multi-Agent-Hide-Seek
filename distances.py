def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def chebyshev(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))
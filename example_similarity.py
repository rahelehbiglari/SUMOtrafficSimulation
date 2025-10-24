import numpy as np

if __name__ == "__main__":
    x = np.array([1, 2, 4, 5, 10, 20, 30]) # One road with one ttraffic light
    y = np.array([1, 2, 5, 4, 10, 20, 30]) # One road with one ttraffic light

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    distance = np.linalg.norm(x - y)
    print(f"norm_x = {norm_x}")
    print(f"norm_y = {norm_y}")
    print(f"distance = {distance}")

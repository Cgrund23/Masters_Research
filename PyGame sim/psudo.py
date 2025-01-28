import numpy as np
import time

for x in range(1,25):
    start = time.time()
    np.linalg.pinv(np.ones((300,300)))
    end = time.time()
    print(end - start)
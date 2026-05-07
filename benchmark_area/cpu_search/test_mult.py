import torch
import pandas as pd
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt

times = []

for i in tqdm.tqdm(range(0, 100_000, 512)):
    keys = torch.randn(1, 24, i, 128)
    query = torch.randn(1, 24, 1, 128)

    start = time.time()
    # result = query * keys
    weights = torch.matmul(query, keys.transpose(2, 3))
    end = time.time()

    times.append(end - start)

times = np.array(times)
plt.plot(times)
plt.savefig("mult_time.png")

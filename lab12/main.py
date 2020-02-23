import numpy as np
import matplotlib.pyplot as plt
import timeit
from google.cloud import storage

FILENAME = "RTS_lab1.txt"
BUCKET_NAME = 'publ_fanout_buck'
OMEGA_MAX = 500
n = 500
N = 1024


def update_blob(data):
    storage_client = storage.Client.from_service_account_json(
        "streamingAccCred.json")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILENAME)
    blob.upload_from_string(data)


def timer(function):
    def new_function(*args, **kwargs):
        start_time = timeit.default_timer()
        val = function(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        return val, elapsed
    return new_function


A = np.random.random_sample(size=n)
fi = np.random.random_sample(size=n) * 2 * np.math.pi
omega = np.linspace(1, n+1, n) * OMEGA_MAX / n
t = np.linspace(0, N-1, N)
x = np.array([0 for _ in t])
x_t = np.vectorize(lambda ti: np.sum(A * np.sin(omega * ti + fi)))
x = x_t(t)

get_Mx = timer(lambda x: np.sum(x) / N)
get_Dx = timer(lambda x: np.sum((x - Mx) ** 2) / (N - 1))
get_Rxx = timer(np.vectorize(lambda tau: np.sum((x[:int(N/2)] - Mx) * (x[int(tau):int(tau)+int(N/2)] - Mx)) / (N/2 - 1)))

Mx, T_Mx = get_Mx(x)
Dx, T_Dx = get_Dx(x)
Rxx, T_Rxx = get_Rxx(t[:int(N/2)])
print(f"     T\nRxx: {T_Rxx}")
# update_blob(f"    Value    T\nMx:{Mx:9f} {T_Mx}\nDx:{Dx:9f} {T_Dx}")
# print(f"https://storage.googleapis.com/publ_fanout_buck/{FILENAME}")

fig = plt.figure(figsize=[8, 6])
plot_x, plot_Rxx = fig.subplots(2, 1)
plot_x.plot(t, x)
plot_x.set_ylabel("x(t)")
plot_x.set_xlabel("t")
plot_Rxx.plot(t[:int(N/2)], Rxx)
plot_Rxx.set_ylabel("Rxx(t)")
plot_Rxx.set_xlabel("t")
fig.show()
plt.show()

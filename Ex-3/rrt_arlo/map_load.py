import time

import matplotlib.pyplot as plt
import numpy as np
import paramiko

from map.aabb import AABB
from map_plot_markers import map_high, map_low

host = "172.20.10.2"
port = 22
username = "pi"
password = "DIKU4Ever"
poll_interval = 1.0

data_remote_path = "/home/pi/Desktop/REX/Ex-3/rrt_arlo/map_data.npy"
data_local_path = "./map_data.npy"

path_remote_path = "/home/pi/Desktop/REX/Ex-3/rrt_arlo/path.npy"
path_local_path = "./path.npy"


def update_map(ax, map_aabb):
    ax.clear()

    try:
        map_data = np.load(data_local_path)
        if map_data.size == 0:
            print("[!] Loaded array is empty — skipping update.")
            return
    except (EOFError, ValueError) as e:
        print(f"[!] Failed to load map data: {e} — skipping update.")
        return
    path_data = np.load(path_local_path)

    plt.imshow(
        map_data.transpose(),
        cmap="Greys",
        origin="lower",
        vmin=0,
        vmax=1,
        extent=(
            map_aabb.left,
            map_aabb.right,
            map_aabb.bottom,
            map_aabb.top,
        ),
        interpolation="none",
    )

    ax.plot(path_data[:, 0], path_data[:, 1], "-r")
    ax.grid(True)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.draw()
    plt.pause(0.01)


def get_remote_file(sftp, remote_path):
    try:
        with sftp.open(remote_path, "rb") as remote_file:
            data = remote_file.read()
            return data
    except IOError as e:
        print(f"[!] Error reading remote file: {e}")
        return None


print(f"[+] Connecting to {host}:{port} via SSH as {username}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username=username, password=password)
sftp = ssh.open_sftp()

map_aabb = AABB(low=map_low, high=map_high)

plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

try:
    while True:
        data = get_remote_file(sftp, data_remote_path)
        path = get_remote_file(sftp, path_remote_path)

        if path == None:
            path = np.array([])

        if data != None:
            with open(data_local_path, "wb") as f:
                f.write(data)
            with open(path_local_path, "wb") as f:
                f.write(path)

            update_map(ax, map_aabb)

        time.sleep(poll_interval)
except KeyboardInterrupt:
    print("\n[!] Interrupted by user. Exiting.")
finally:
    sftp.close()
    ssh.close()
    print("[+] SSH connection closed.")

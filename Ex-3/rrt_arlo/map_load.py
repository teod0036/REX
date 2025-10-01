import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import paramiko

from map.aabb import AABB
from map_plot_markers import map_high, map_low


def update_map(fp, ax, map_aabb):
    ax.clear()

    try:
        map_data = np.load(fp)
        if map_data.size == 0:
            print("[!] Loaded array is empty — skipping update.")
            return
    except (EOFError, ValueError) as e:
        print(f"[!] Failed to load map data: {e} — skipping update.")
        return

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

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.draw()
    plt.pause(0.01)


def live_plot_loop(
    host, port, username, password, remote_path, local_path, poll_interval=1.0
):
    def get_remote_file_hash(sftp, remote_path):
        try:
            with sftp.open(remote_path, "rb") as remote_file:
                data = remote_file.read()
                return hashlib.md5(data).hexdigest(), data
        except IOError as e:
            print(f"[!] Error reading remote file: {e}")
            return None, None

    print(f"[+] Connecting to {host}:{port} via SSH as {username}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=username, password=password)
    sftp = ssh.open_sftp()

    map_aabb = AABB(low=map_low, high=map_high)

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()
    last_hash = None

    try:
        while True:
            current_hash, data = get_remote_file_hash(sftp, remote_path)
            if data != None and current_hash and current_hash != last_hash:
                with open(local_path, "wb") as f:
                    f.write(data)
                last_hash = current_hash
                print(f"[✓] Updated local file: {local_path}")

                update_map(local_path, ax, map_aabb)

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Exiting.")
    finally:
        sftp.close()
        ssh.close()
        print("[+] SSH connection closed.")


if __name__ == "__main__":
    live_plot_loop(
        host="172.20.10.2",
        port=22,
        username="pi",
        password="DIKU4Ever",
        remote_path="/home/pi/Desktop/REX/Ex-3/rrt_arlo/map_data.npy",
        local_path="./map_data.npy",
        poll_interval=1.0,
    )

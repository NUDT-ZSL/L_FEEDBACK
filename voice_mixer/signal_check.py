"""Ad-hoc real SIGINT / resume-rejection verification driver."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time


def main():
    root = os.path.join(os.getcwd(), "tmp_sig")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    with open(os.path.join("out", "sample_project.json"), encoding="utf-8") as handle:
        project = json.load(handle)
    project["export"]["output_wav"] = os.path.join(root, "r.wav")
    project["export"]["manifest"] = os.path.join(root, "m.json")
    project["export"]["work_dir"] = os.path.join(root, "work")
    project["export"]["chunk_samples"] = 1000
    project_path = os.path.join(root, "p.json")
    with open(project_path, "w", encoding="utf-8") as handle:
        json.dump(project, handle)

    proc = subprocess.Popen(
        [sys.executable, "main.py", project_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.15)
    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=10)
    print("first exit code:", proc.returncode)
    print("snapshot exists:", os.path.isfile(os.path.join(root, "work", "snapshot.json")))
    print("wav absent:", not os.path.exists(os.path.join(root, "r.wav")))

    resumed = subprocess.run(
        [sys.executable, "main.py", project_path], capture_output=True, text=True
    )
    print("resume exit code:", resumed.returncode)
    print("wav present:", os.path.exists(os.path.join(root, "r.wav")))

    project["events"][0]["tick"] = 101
    with open(project_path, "w", encoding="utf-8") as handle:
        json.dump(project, handle)
    changed = subprocess.run(
        [sys.executable, "main.py", project_path], capture_output=True, text=True
    )
    print("changed-project exit code:", changed.returncode)
    with open(os.path.join(root, "m.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    print("resume rejected flag:", manifest["resume"]["rejected"])
    print("status:", manifest["status"])


if __name__ == "__main__":
    main()

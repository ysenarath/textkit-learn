import platform
import re
import shutil
import subprocess
import sys

cuda_version_pattern = re.compile(r"CUDA Version:\s+(\d+)\.(\d+)")


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        match = cuda_version_pattern.search(output)
        if match:
            major, minor = match.groups()
            return f"cu{major}{minor}"
    except Exception:
        return None
    return None


def auto_install():
    manager = "uv" if shutil.which("uv") else "pip"
    system = platform.system()

    # Logic to select the right URL/Extra
    if system == "Darwin":
        target = "metal"
    else:
        cuda = get_cuda_version()
        target = cuda if cuda else "cpu"

    url = f"https://abetlen.github.io/llama-cpp-python/whl/{target}"

    # The final command
    if manager == "uv":
        cmd = [
            "uv",
            "pip",
            "install",
            "--upgrade",
            "llama-cpp-python",
            "--extra-index-url",
            url,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "llama-cpp-python",
            "--extra-index-url",
            url,
        ]

    print(
        f"🚀 Using {manager} to install llama-cpp-python for {target} with '{' '.join(cmd)}'..."
    )
    subprocess.check_call(cmd)


if __name__ == "__main__":
    auto_install()

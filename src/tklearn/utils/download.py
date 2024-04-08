import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Union

import requests
from tqdm import auto as tqdm

try:
    import wget
except ImportError as e:
    wget_error_message = str(e)
    wget = None


__all__ = [
    "download",
]


def download_(url: str, path: Union[str, Path], verbose: bool = False) -> None:
    response = requests.get(url, stream=True)
    block_size = 1024  # 1 Kibibyte
    progress_bar = None
    total_size_in_bytes = 0
    if verbose:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm.tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading",
        )
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as df:
        if progress_bar is None:
            shutil.copyfileobj(response.raw, df)
        else:
            for data in response.iter_content(block_size):
                if progress_bar:
                    progress_bar.update(len(data))
                df.write(data)
    response.close()
    if progress_bar:
        progress_bar.close()
    if (
        progress_bar
        and total_size_in_bytes != 0
        and progress_bar.n != total_size_in_bytes
    ):
        raise ValueError("something went wrong")


def download(
    url: str,
    path: Union[str, Path],
    verbose: bool = False,
    force: bool = False,
    exist_ok: bool = False,
    unzip: bool = True,
    mode: str = "auto",
) -> None:
    if force or not path.exists():
        mode = mode.lower()
        if (
            "wget" if mode == "auto" and not verbose else mode
        ) == "wget" and wget is not None:
            wget.download(url, str(path))
        elif mode == "wget":
            raise ImportError(wget_error_message)
        else:
            download_(url, path, verbose)
    if not unzip:
        if exist_ok:
            return
        raise FileExistsError(f"'{path}' already exists")
    unzip_path = path.with_suffix("")
    if unzip_path.exists():
        if exist_ok:
            return
        raise FileExistsError(f"'{unzip_path}' already exists")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as f_in, open(unzip_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"unsupported file extension: {path.suffix}")

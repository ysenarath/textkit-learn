import os
from pathlib import Path

import click

cwd = Path(__file__).parent


@click.group()
def main():
    pass


@main.command()
@click.option("--make", is_flag=True, help="make docs")
@click.option("--open", is_flag=True, help="open docs")
def docs(make, open):
    if make:
        os.system(f"rm -rf {cwd}/docs/html/")
        os.system(f"rm -rf {cwd}/docs/doctrees/")
        os.system(f"sphinx-apidoc -o {cwd}/docs/source/api {cwd}/src/")
        os.system(f"sphinx-build -M html {cwd}/docs/source/ {cwd}/docs/")
    if open:
        index = f"{cwd}/docs/html/index.html"
        os.system(f"open {index}")


@main.command()
@click.argument("name", default="html_basic_demo")
def run(name: str):
    # runs demo
    os.system(f"uv run {cwd}/examples/{name}.py")


if __name__ == "__main__":
    main()

import click

from tklearn import config
from tklearn.kb.conceptnet.io import ConceptNetIO


@click.group()
def cli():
    pass


@cli.group("conceptnet")
def cli_conceptnet():
    pass


@cli_conceptnet.command("download")
def cli_conceptnet_download():
    url = config.external.conceptnet.download_url
    output_dir = config.cache_dir / "resources" / "conceptnet"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    path = output_dir / filename
    io = ConceptNetIO(path)
    io.download(url=url, verbose=True, unzip=True, exist_ok=True)
    convert_to_jsonld = not io.jsonld_path.exists()
    convert_to_jsonl = not io.jsonl_path.exists()
    if convert_to_jsonld or convert_to_jsonl:
        io.read_csv(verbose=True)
    if convert_to_jsonld:
        io.to_jsonld()
    if convert_to_jsonl:
        io.to_jsonl()


if __name__ == "__main__":
    cli()

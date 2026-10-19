"""``tklearn kb``: look up words, find them in text and augment text with
a knowledge base, by default `tklearn.kb.WiktionaryStore`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from tklearn.kb import KnowledgeBase

__all__ = [
    "kb",
]


@click.group()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of a TableStore. By default, the Wiktionary store, "
    "downloaded the first time.",
)
@click.pass_context
def kb(ctx: click.Context, path: Path | None) -> None:
    """Query a knowledge base, by default English Wiktionary."""
    ctx.obj = path


@kb.command()
@click.argument("word")
@click.pass_obj
def senses(path: Path | None, word: str) -> None:
    """List the senses of WORD: their ids and glosses."""
    knowledge_base = _knowledge_base(path)
    for candidate in knowledge_base.extract_candidates(word):
        click.echo(f"#{candidate.sense_id}\t{candidate.definition}")


@kb.command()
@click.argument("word")
@click.option(
    "-s", "--sense", type=int, help="Only this sense; by default all."
)
@click.option(
    "-p",
    "--predicate",
    "predicates",
    multiple=True,
    metavar="NAME",
    help="Only this relation, e.g. synonym; repeat for more. By default all.",
)
@click.pass_obj
def relations(
    path: Path | None,
    word: str,
    sense: int | None,
    predicates: tuple[str, ...],
) -> None:
    """List the relations of WORD, as subject, predicate and object.

    A word#id is one sense of the word; a word alone is the word in
    general.
    """
    knowledge_base = _knowledge_base(path)
    word_sense = word if sense is None else (word, sense)
    seen = set()
    for candidate in knowledge_base.extract_candidates(word_sense):
        for triple in knowledge_base.extract_relations(
            candidate, predicates or None
        ):
            # relations of the word in general repeat for each sense
            if triple in seen:
                continue
            seen.add(triple)
            subject, predicate, object_ = triple
            click.echo(
                f"{_word_sense(subject)}\t{predicate}\t{_word_sense(object_)}"
            )


@kb.command()
@click.argument("text")
@click.pass_obj
def mentions(path: Path | None, text: str) -> None:
    """Find the known words in TEXT, with the senses each may have."""
    knowledge_base = _knowledge_base(path)
    for mention in knowledge_base.extract_mentions(text):
        start, end = mention.span
        click.echo(f"{mention.form} ({start}-{end})")
        for candidate in mention.candidates:
            word_sense = (candidate.word, candidate.sense_id)
            click.echo(f"  {_word_sense(word_sense)}\t{candidate.definition}")


@kb.command()
@click.argument("text")
@click.option(
    "-p",
    "--predicate",
    "predicates",
    multiple=True,
    default=("synonym",),
    show_default=True,
    metavar="NAME",
    help="Relation that finds replacements; repeat for more.",
)
@click.pass_obj
def augment(path: Path | None, text: str, predicates: tuple[str, ...]) -> None:
    """Print variants of TEXT with one word replaced by a related one."""
    knowledge_base = _knowledge_base(path)
    for augmentation in knowledge_base.augment(text, predicates=predicates):
        click.echo(augmentation.text)


def _knowledge_base(path: Path | None) -> KnowledgeBase:
    # imported here, so that `tklearn --help` does not load the kb
    from tklearn.kb import KnowledgeBase, TableStore, WiktionaryStore

    store = WiktionaryStore() if path is None else TableStore(path)
    return KnowledgeBase(store)


def _word_sense(word_sense: tuple[str, int | None]) -> str:
    word, sense_id = word_sense
    return word if sense_id is None else f"{word}#{sense_id}"

from __future__ import annotations

from typing import Iterator, List, Optional

import orjson
from pydantic import BaseModel, Field


class ExtraData(BaseModel):
    words: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class InfoTemplate(BaseModel):
    name: str
    extra_data: ExtraData
    expansion: str


class Example(BaseModel):
    text: Optional[str] = None
    type: Optional[str] = None
    ref: Optional[str] = None
    english: Optional[str] = None
    roman: Optional[str] = None
    tags: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    ruby: Optional[List[List[str]]] = None
    note: Optional[str] = None
    literal_meaning: Optional[str] = None


class Synonym(BaseModel):
    word: str
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    english: Optional[str] = None
    sense: Optional[str] = None
    alt: Optional[str] = None
    topics: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    roman: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    taxonomic: Optional[str] = None
    qualifier: Optional[str] = None
    urls: Optional[List[str]] = None
    extra: Optional[str] = None


class Hyponym(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    taxonomic: Optional[str] = None
    alt: Optional[str] = None
    english: Optional[str] = None
    sense: Optional[str] = None
    topics: Optional[List[str]] = None
    qualifier: Optional[str] = None
    roman: Optional[str] = None
    urls: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    ruby: Optional[List[List[str]]] = None


class Antonym(BaseModel):
    word: str
    source: Optional[str] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    english: Optional[str] = None
    alt: Optional[str] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    urls: Optional[List[str]] = None


class FormOfItem(BaseModel):
    word: str
    extra: Optional[str] = None


class CompoundOfItem(BaseModel):
    word: str
    extra: Optional[str] = None


class Hypernym(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    sense: Optional[str] = None
    english: Optional[str] = None
    topics: Optional[List[str]] = None
    roman: Optional[str] = None
    raw_tags: Optional[List[str]] = None
    ruby: Optional[List[List[str]]] = None
    alt: Optional[str] = None
    taxonomic: Optional[str] = None
    urls: Optional[List[str]] = None


class Meronym(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    sense: Optional[str] = None
    topics: Optional[List[str]] = None
    english: Optional[str] = None
    roman: Optional[str] = None
    alt: Optional[str] = None
    urls: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    ruby: Optional[List[List[str]]] = None
    taxonomic: Optional[str] = None


class RelatedItem(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    english: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    alt: Optional[str] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    topics: Optional[List[str]] = None
    urls: Optional[List[str]] = None
    qualifier: Optional[str] = None
    raw_tags: Optional[List[str]] = None
    taxonomic: Optional[str] = None


class Instance(BaseModel):
    word: str
    source: str
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    source: Optional[str] = None


class CoordinateTerm(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sense: Optional[str] = None
    english: Optional[str] = None
    roman: Optional[str] = None
    alt: Optional[str] = None
    urls: Optional[List[str]] = None
    ruby: Optional[List[List[str]]] = None
    raw_tags: Optional[List[str]] = None
    taxonomic: Optional[str] = None
    qualifier: Optional[str] = None


class Holonym(BaseModel):
    word: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    topics: Optional[List[str]] = None
    english: Optional[str] = None
    alt: Optional[str] = None
    qualifier: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    raw_tags: Optional[List[str]] = None


class Sense(BaseModel):
    examples: Optional[List[Example]] = None
    wikidata: Optional[List[str]] = None
    senseid: Optional[List[str]] = None
    links: Optional[List[List[str]]] = None
    synonyms: Optional[List[Synonym]] = None
    categories: Optional[List[str]] = None
    glosses: Optional[List[str]] = None
    raw_glosses: Optional[List[str]] = None
    qualifier: Optional[str] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    hyponyms: Optional[List[Hyponym]] = None
    antonyms: Optional[List[Antonym]] = None
    raw_tags: Optional[List[str]] = None
    info_templates: Optional[List[InfoTemplate]] = None
    alt_of: Optional[List[AltOfItem]] = None
    form_of: Optional[List[FormOfItem]] = None
    wikipedia: Optional[List[str]] = None
    head_nr: Optional[int] = None
    taxonomic: Optional[str] = None
    compound_of: Optional[List[CompoundOfItem]] = None
    hypernyms: Optional[List[Hypernym]] = None
    meronyms: Optional[List[Meronym]] = None
    related: Optional[List[RelatedItem]] = None
    instances: Optional[List[Instance]] = None
    coordinate_terms: Optional[List[CoordinateTerm]] = None
    holonyms: Optional[List[Holonym]] = None


class HeadTemplate(BaseModel):
    name: str
    expansion: str


class Form(BaseModel):
    form: str
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    ipa: Optional[str] = None
    roman: Optional[str] = None
    raw_tags: Optional[List[str]] = None
    head_nr: Optional[int] = None
    topics: Optional[List[str]] = None


class DerivedItem(BaseModel):
    word: str
    english: Optional[str] = None
    taxonomic: Optional[str] = None
    alt: Optional[str] = None
    tags: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    urls: Optional[List[str]] = None
    qualifier: Optional[str] = None


class Translation(BaseModel):
    lang: str
    code: Optional[str] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    word: Optional[str] = None
    tags: Optional[List[str]] = None
    note: Optional[str] = None
    english: Optional[str] = None
    alt: Optional[str] = None
    topics: Optional[List[str]] = None
    raw_tags: Optional[List[str]] = None
    taxonomic: Optional[str] = None


class Sound(BaseModel):
    tags: Optional[List[str]] = None
    ipa: Optional[str] = None
    audio: Optional[str] = None
    ogg_url: Optional[str] = None
    mp3_url: Optional[str] = None
    enpr: Optional[str] = None
    rhymes: Optional[str] = None
    homophone: Optional[str] = None
    note: Optional[str] = None
    zh_pron: Optional[str] = Field(None, alias="zh-pron")
    other: Optional[str] = None
    text: Optional[str] = None
    hangeul: Optional[str] = None
    topics: Optional[List[str]] = None
    form: Optional[str] = None
    audio_ipa: Optional[str] = Field(None, alias="audio-ipa")


class EtymologyTemplate(BaseModel):
    name: str
    expansion: str


class InflectionTemplate(BaseModel):
    name: str


class Template(BaseModel):
    name: str
    expansion: str


class Descendant(BaseModel):
    # depth: int
    # templates: List[Template]
    # text: str
    depth: Optional[int] = None
    templates: Optional[List[Template]] = Field(default_factory=list)
    text: Optional[str] = None


class Abbreviation(BaseModel):
    sense: Optional[str] = None
    word: str
    alt: Optional[str] = None
    urls: Optional[List[str]] = None
    english: Optional[str] = None
    topics: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    roman: Optional[str] = None


class Proverb(BaseModel):
    roman: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    english: Optional[str] = None
    word: str
    tags: Optional[List[str]] = None
    alt: Optional[str] = None


class Troponym(BaseModel):
    word: str
    source: Optional[str] = None
    sense: Optional[str] = None
    roman: Optional[str] = None
    ruby: Optional[List[List[str]]] = None
    english: Optional[str] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None


class EtymologyExample(BaseModel):
    raw_tags: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    ref: Optional[str] = None
    roman: Optional[str] = None
    text: str
    type: str
    english: Optional[str] = None


class AltOfItem(BaseModel):
    word: str
    extra: Optional[str] = None


class Word(BaseModel):
    word: Optional[str] = None
    senses: Optional[List[Sense]] = None
    pos: Optional[str] = None
    head_templates: Optional[List[HeadTemplate]] = None
    categories: Optional[List[str]] = None
    forms: Optional[List[Form]] = None
    synonyms: Optional[List[Synonym]] = None
    hyponyms: Optional[List[Hyponym]] = None
    derived: Optional[List[DerivedItem]] = None
    related: Optional[List[RelatedItem]] = None
    translations: Optional[List[Translation]] = None
    sounds: Optional[List[Sound]] = None
    hyphenation: Optional[List[str]] = None
    etymology_text: Optional[str] = None
    etymology_templates: Optional[List[EtymologyTemplate]] = None
    lang: Optional[str] = None
    lang_code: Optional[str] = None
    hypernyms: Optional[List[Hypernym]] = None
    coordinate_terms: Optional[List[CoordinateTerm]] = None
    meronyms: Optional[List[Meronym]] = None
    antonyms: Optional[List[Antonym]] = None
    inflection_templates: Optional[List[InflectionTemplate]] = None
    wikipedia: Optional[List[str]] = None
    descendants: Optional[List[Descendant]] = None
    etymology_number: Optional[int] = None
    holonyms: Optional[List[Holonym]] = None
    original_title: Optional[str] = None
    instances: Optional[List[Instance]] = None
    title: Optional[str] = None
    redirect: Optional[str] = None
    abbreviations: Optional[List[Abbreviation]] = None
    info_templates: Optional[List[InfoTemplate]] = None
    redirects: Optional[List[str]] = None
    proverbs: Optional[List[Proverb]] = None
    form_of: Optional[List[FormOfItem]] = None
    troponyms: Optional[List[Troponym]] = None
    etymology_examples: Optional[List[EtymologyExample]] = None
    literal_meaning: Optional[str] = None
    topics: Optional[List[str]] = None
    alt_of: Optional[List[AltOfItem]] = None
    wikidata: Optional[List[str]] = None
    source: Optional[str] = None


def parse_jsonl(file_path: str) -> Iterator[Word]:
    """
    Parses a JSONL file and yields validated Word objects.

    Parameters
    ----------
    file_path : str
        Path to the JSONL file.

    Yields
    ------
    Word
        Parsed and validated Word object.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = orjson.loads(line)
            yield Word.model_validate(data)


# RelationTypesWithSense = [
#     Synonym, - same meaning
#     Antonym, - opposite meaning
#     Hyponym, - subtype of
#     Hypernym, - type of
#     Meronym, - part of
#     Holonym, - whole of
#     RelatedItem, - related
#     Instance, - instance of
#     CoordinateTerm, - coordinate term
#     DerivedItem,
#     Translation,
#     Abbreviation,
#     Troponym,
# ]

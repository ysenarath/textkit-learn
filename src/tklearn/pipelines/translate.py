from __future__ import annotations

import shutil
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    overload,
)

from datasets import Dataset, DatasetDict
from nightjar import AutoModule, BaseConfig, BaseModule
from tqdm import auto as tqdm
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from tklearn.utils.datasets import map_dataset

__all__ = [
    "TranslationConfig",
    "TranslationPipeline",
    "GoogleMadlad4003BMTPipeline",
    "FacebookMBARTLargeM2MPipeline",
]


class TranslationConfig(
    BaseConfig, dispatch=["pretrained_model_name_or_path"]
):
    pretrained_model_name_or_path: ClassVar[str]
    input_max_length: int = 125
    max_new_tokens: int = 125


class TranslationPipeline(BaseModule):
    config: TranslationConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @overload
    def __call__(
        self,
        text: str,
        *,
        tgt_lang: str = None,
        src_lang: Optional[str] = None,
    ) -> str:
        pass

    @overload
    def __call__(
        self,
        text: Iterable[str],
        *,
        tgt_lang: str = None,
        src_lang: Optional[str] = None,
    ) -> List[str]:
        pass

    def __init__(self, config: Optional[TranslationConfig] = None) -> None: ...

    def __call__(
        self,
        text: Iterable[str] | str,
        *,
        tgt_lang: str = None,
        src_lang: Optional[str] = None,
    ) -> str | List[str]:
        raise NotImplementedError


class AutoTranslationPipeline(AutoModule):
    def __new__(cls, config: TranslationConfig) -> TranslationPipeline:
        pipeline = super().__new__(cls, config)
        if not isinstance(pipeline, TranslationPipeline):
            msg = (
                f"expected {TranslationPipeline.__name__}, "
                f"got {pipeline.__class__.__name__}"
            )
            raise TypeError(msg)
        return pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        input_max_length: int = 125,
        max_new_tokens: int = 125,
        **kwargs: Any,
    ) -> TranslationPipeline:
        kwargs.update(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            input_max_length=input_max_length,
            max_new_tokens=max_new_tokens,
        )
        config = TranslationConfig.from_dict(kwargs)
        return cls(config)


GOOGLE_MADLAD_400_3B_MT_LANG_CODE_TO_LANG_ID = {
    "ar": "ar",
    "en": "en",
    "zh": "zh",
    "fr": "fr",
    "de": "de",
    "ru": "ru",
    "tr": "tr",
    "hi": "hi",
    "ko": "ko",
    "it": "it",
    "es": "es",
    "pt": "pt",
    "id": "id",
    "no": "no",
}


class GoogleMadlad4003BMTPipelineConfig(TranslationConfig):
    pretrained_model_name_or_path: ClassVar[str] = "google/madlad400-3b-mt"


class GoogleMadlad4003BMTPipeline(TranslationPipeline):
    config: GoogleMadlad4003BMTPipelineConfig

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        if config is None:
            config = TranslationConfig()
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.pretrained_model_name_or_path, device_map="auto"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
        )

    def __call__(
        self,
        text: Iterable[str] | str,
        *,
        tgt_lang: str = None,
        src_lang: Optional[str] = None,
    ) -> str | List[str]:
        if tgt_lang in GOOGLE_MADLAD_400_3B_MT_LANG_CODE_TO_LANG_ID:
            tgt_lang = GOOGLE_MADLAD_400_3B_MT_LANG_CODE_TO_LANG_ID[tgt_lang]
        output_first = isinstance(text, str)
        if output_first:
            text = [text]
        text = [f"<2{tgt_lang}> {t}" for t in text]
        kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": self.config.input_max_length,
        }
        input_encodings = self.tokenizer(text, **kwargs)
        input_ids = input_encodings.input_ids.to(self.model.device)
        attention_mask = input_encodings.attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
        )
        decoded = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        if output_first:
            return decoded[0]
        return decoded


FB_MBART_LANG_CODE_TO_LANG_ID = {
    "ar": "ar_AR",
    "en": "en_XX",
    "zh": "zh_CN",
    "fr": "fr_XX",
    "de": "de_DE",
    "ru": "ru_RU",
    "tr": "tr_TR",
    "hi": "hi_IN",
    "ko": "ko_KR",
    "it": "it_IT",
    "es": "es_XX",
    "pt": "pt_XX",
    "id": "id_ID",
}


class FacebookMBARTLargeM2MPipelineConfig(TranslationConfig):
    pretrained_model_name_or_path: ClassVar[str] = (
        "facebook/mbart-large-50-many-to-many-mmt"
    )


class FacebookMBARTLargeM2MPipeline(TranslationPipeline):
    config: FacebookMBARTLargeM2MPipelineConfig

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        if config is None:
            config = TranslationConfig()
        self.config = config
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.config.pretrained_model_name_or_path, device_map="auto"
        )
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.config.pretrained_model_name_or_path,
        )

    def __call__(
        self,
        text: Iterable[str] | str,
        *,
        tgt_lang: str = None,
        src_lang: Optional[str] = None,
    ) -> str | List[str]:
        if src_lang in FB_MBART_LANG_CODE_TO_LANG_ID:
            src_lang = FB_MBART_LANG_CODE_TO_LANG_ID[src_lang]
        if tgt_lang in FB_MBART_LANG_CODE_TO_LANG_ID:
            tgt_lang = FB_MBART_LANG_CODE_TO_LANG_ID[tgt_lang]
        self.tokenizer.src_lang = src_lang
        output_first = isinstance(text, str)
        if output_first:
            text = [text]
        encoded_hi = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_hi,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
        )
        decoded = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        if output_first:
            return decoded[0]
        return decoded


def _translate_dataset_batch(
    batch: Any,
    *,
    translate: Callable[[str, str, str | None], str],
    tgt_lang: str,
    src_lang: str | None = None,
    text_col: str = "text",
) -> Dataset | DatasetDict:
    # from hatedetect.translate import TranslationConfig, TranslationPipeline
    batch[text_col] = translate(
        batch[text_col], tgt_lang=tgt_lang, src_lang=src_lang
    )
    batch["lang"] = [tgt_lang] * len(batch[text_col])
    return batch


def translate_dataset(
    dataset: Dataset | DatasetDict,
    translate: Any,
    tgt_lang: str,
    src_lang: str | None = None,
    text_col: str = "text",
    batch_size: int = 1,
    verbose: bool = True,
) -> Dataset | DatasetDict:
    return map_dataset(
        dataset,
        _translate_dataset_batch,
        batch_size=batch_size,
        verbose=verbose,
        func_kwargs={
            "translate": translate,
            "tgt_lang": tgt_lang,
            "src_lang": src_lang,
            "text_col": text_col,
        },
    )


def _langs_extend_exclude_util(
    langs: List[str] | str,
    extend_langs: List[str] | str | None = None,
    exclude_langs: List[str] | str | None = None,
) -> List[str]:
    if isinstance(langs, str):
        langs = [s.strip() for s in langs.split(",")]
    langs: set = set(langs)
    if exclude_langs == "all":
        exclude_langs = langs.copy()
    if extend_langs is not None:
        if isinstance(extend_langs, str):
            extend_langs = [s.strip() for s in extend_langs.split(",")]
        langs.update(extend_langs)
    if exclude_langs is not None:
        if isinstance(exclude_langs, str):
            exclude_langs = [s.strip() for s in exclude_langs.split(",")]
        langs.difference_update(exclude_langs)
    return list(langs)


def cross_translate_dataset(
    dataset: Dataset | DatasetDict,
    translate: Any,
    lang_col: str = "lang",
    text_col: str = "text",
    extend_tgt_langs: List[str] | str | None = None,
    exclude_tgt_langs: List[str] | str | None = None,
    exclude_src_langs: List[str] | str | None = None,
    batch_size: int = 1,
    verbose: bool = True,
    save_dir: Path | str | None = None,
) -> Dict[str, Dataset | DatasetDict]:
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir is not None:
        # create the save_dir if it does not exist
        save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, Dataset):
        languages = set()
        languages.update(dataset.unique(lang_col))
    elif isinstance(dataset, DatasetDict):
        languages = set()
        for key in dataset.keys():
            languages.update(dataset[key].unique(lang_col))
    else:
        msg = "dataset must be of type Dataset or DatasetDict"
        raise ValueError(msg)
    final_datasets = {}
    nlang = len(languages)
    lang_pbar = tqdm.tqdm(
        languages,
        desc="Translating Datasets",
        disable=not verbose,
        total=nlang * nlang,
    )
    src_langs = _langs_extend_exclude_util(
        languages,
        exclude_langs=exclude_src_langs,
    )
    tgt_langs = _langs_extend_exclude_util(
        languages,
        extend_langs=extend_tgt_langs,
        exclude_langs=exclude_tgt_langs,
    )
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            src_lang_dataset = dataset.filter(
                lambda x: x[lang_col] == src_lang
            )
            filename = f"{src_lang}-{tgt_lang}"
            lang_pbar.set_postfix_str(filename)
            tgt_lang_dataset = None
            if save_dir and (save_dir / filename).exists():
                if isinstance(src_lang_dataset, DatasetDict):
                    tgt_lang_dataset = DatasetDict.load_from_disk(
                        save_dir / filename
                    )
                    # check lengths of datasets
                    if set(tgt_lang_dataset.keys()) != set(
                        src_lang_dataset.keys()
                    ):
                        tgt_lang_dataset = None
                    for key in tgt_lang_dataset.keys():
                        if len(tgt_lang_dataset[key]) == len(
                            src_lang_dataset[key]
                        ):
                            # saved dataset is valid (i think)
                            #   reson: length of datasets are same
                            continue
                        tgt_lang_dataset = None
                        break
                else:
                    tgt_lang_dataset = Dataset.load_from_disk(
                        save_dir / filename
                    )
                    if len(tgt_lang_dataset) != len(src_lang_dataset):
                        tgt_lang_dataset = None
                if tgt_lang_dataset is None:
                    # remove the existing dataset
                    shutil.rmtree(save_dir / filename)
            if tgt_lang_dataset is not None:
                pass  # use the existing dataset
            elif src_lang == tgt_lang:
                # use the src_lang_dataset as the tgt_lang_dataset
                tgt_lang_dataset = src_lang_dataset
            else:
                tgt_lang_dataset = translate_dataset(
                    src_lang_dataset,
                    translate,
                    tgt_lang,
                    src_lang,
                    text_col,
                    batch_size,
                    verbose,
                )
            if save_dir and not (save_dir / filename).exists():
                # save the tgt_lang_dataset to save_dir with a directory
                # name that includes the src_lang and tgt_lang
                tgt_lang_dataset.save_to_disk(save_dir / filename)
            final_datasets[(src_lang, tgt_lang)] = tgt_lang_dataset
            lang_pbar.update(1)
    return final_datasets

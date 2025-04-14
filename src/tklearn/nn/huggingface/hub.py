import json
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import ModelCard, upload_folder
from transformers.modelcard import TrainingSummary
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from tklearn import config, logging
from tklearn.nn.models.base import Model

logger = logging.get_logger(__name__)


hf_models_dir = config.assets_dir / "huggingface" / "transformers"
hf_datasets_dir = config.assets_dir / "huggingface" / "datasets"
temp_dir = config.temp_dir


class ModelRepo:
    def __init__(
        self, repo_id_or_path: Union[str, Path], path: Optional[Path] = None
    ):
        if path is None:
            path = Path(repo_id_or_path)
            if path.exists():
                self.path = path
                self.repo_id = None
            else:
                # repo_id_or_path is a repo_id with format "username/repo_id"
                org_name, repo_name = repo_id_or_path.split("/")
                self.path = hf_models_dir / org_name / repo_name
                self.repo_id = repo_id_or_path
        else:
            self.repo_id = repo_id_or_path
            self.path = path

    def add_model(
        self,
        config_name: str,
        model: Model,  # not a huggerface model (just torch.nn.Module)
        model_card: Optional[ModelCard] = None,
        training_summary: Optional[TrainingSummary] = None,
    ) -> None:
        # there is no save_pretrained function
        # so we need to save the model manually
        model_dir = self.path / config_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(model.config.to_dict(), f)
        with open(model_dir / "model_state_dict.bin", "wb") as f:
            torch.save(model.state_dict(), f)
        if model_card is not None:
            model_card.save(model_dir / "README.md")
        else:
            with open(model_dir / "README.md", "w") as f:
                f.write("# Model Card\n")
        if training_summary is not None:
            model_card = training_summary.to_model_card()

    def push_to_hub(
        self,
        commit_message: Optional[str] = "Update model",
        blocking: bool = True,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> str:
        model_name = kwargs.pop("model_name", None)
        if model_name is None:
            model_name = self.path.name
        return upload_folder(
            repo_id=self.repo_id,
            folder_path=self.path,
            commit_message=commit_message,
            token=token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
            revision=revision,
        )

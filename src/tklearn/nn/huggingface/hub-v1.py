import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import ModelCard, create_repo, upload_folder
from transformers.modelcard import TrainingSummary

from tklearn import config, logging
from tklearn.nn.models.base import AutoModel, Model

logger = logging.get_logger(__name__)


hf_models_dir = config.assets_dir / "huggingface" / "transformers"
hf_datasets_dir = config.assets_dir / "huggingface" / "datasets"
temp_dir = config.temp_dir


@dataclass
class TrainerConfig:
    output_dir: str = field(default="output")
    hub_model_id: Optional[str] = field(default=None)
    hub_token: Optional[str] = field(default=None)
    should_save: bool = field(default=True)
    hub_private_repo: bool = field(default=True)


class HuggingFaceHub:
    config: TrainerConfig

    def __init__(
        self, config: TrainerConfig, repo_id_or_path: Union[str, Path]
    ):
        path = Path(repo_id_or_path)
        self.path = None
        self.repo_id = None
        if not path.exists():
            self.repo_id = repo_id_or_path
        else:
            self.path = path
        self.config: TrainerConfig = None

    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> str:
        model_name = kwargs.pop("model_name", None)
        if model_name is None and self.config.should_save:
            if self.config.hub_model_id is None:
                model_name = Path(self.config.output_dir).name
            else:
                model_name = self.config.hub_model_id.split("/")[-1]
        token = token if token is not None else self.config.hub_token

        # In case the user calls this method with args.push_to_hub = False
        if self.hub_model_id is None:
            self.init_hf_repo(token=token)

        # Needs to be executed on all processes for TPU training,
        # but will only save on the processed determined by
        # self.args.should_save.
        self.save_model(_internal_call=True)

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        # Add additional tags in the case the model has already some
        # tags and users pass "tags" argument to `push_to_hub` so that
        # trainer automatically handles internal tags from all models
        # since Trainer does not call `model.push_to_hub`.
        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            # If it is a string, convert it to a list
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        # Wait for the current upload to be finished.
        self._finish_current_push()
        return upload_folder(
            repo_id=self.hub_model_id,
            folder_path=self.config.output_dir,
            commit_message=commit_message,
            token=token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
            revision=revision,
        )

    def _finish_current_push(self):
        if not hasattr(self, "push_in_progress"):
            return
        if (
            self.push_in_progress is not None
            and not self.push_in_progress.is_done()
        ):
            logger.info(
                "Waiting for the current checkpoint push to be finished, this might take a couple of minutes."
            )
            self.push_in_progress.wait_until_done()

    def init_hf_repo(self, token: Optional[str] = None):
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        # Only on process zero
        if not self.is_world_process_zero():
            return

        if self.config.hub_model_id is None:
            repo_name = Path(self.config.output_dir).absolute().name
        else:
            repo_name = self.config.hub_model_id

        token = token if token is not None else self.config.hub_token
        repo_url = create_repo(
            repo_name,
            token=token,
            private=self.config.hub_private_repo,
            exist_ok=True,
        )
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None

    def is_world_process_zero(self) -> bool:
        return True

    def create_model_card(
        self,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Union[str, list[str], None] = None,
        dataset_tags: Union[str, list[str], None] = None,
        dataset: Union[str, list[str], None] = None,
        dataset_args: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        model_card_filepath = os.path.join(self.config.output_dir, "README.md")
        if os.path.exists(model_card_filepath):
            mc = ModelCard.load(model_card_filepath)
            library_name = mc.data.get("library_name")
            # Append existing tags in `tags`
            existing_tags = mc.data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        training_summary = TrainingSummary(
            model_name=model_name,
            language=language,
            license=license,
            tags=tags,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset=dataset,
            dataset_tags=dataset_tags,
            dataset_args=dataset_args,
            dataset_metadata=None,
            eval_results=None,
            eval_lines=None,
            hyperparameters=None,
            source="trainer",
        )
        model_card = training_summary.to_model_card()
        with open(model_card_filepath, "w") as f:
            f.write(model_card)


# For models we need to get information from several places if possible
# 1. The model config (model)
# 2. The model parameters (model)
# 3. Trainer configuration (summary)
# 4. The training summary (history)


class ModelRepo:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path)

    def load(self) -> Model:
        with open("config.json", "r") as f:
            config = json.load(f)
        model = AutoModel(config)
        state_dict = torch.load("model.pth")
        model.load_state_dict(state_dict)
        return model

    def save(self, model: Model):
        with open("config.json", "w") as f:
            json.dump(model.config, f)
        torch.save(model.state_dict(), "model.pth")

    def push_to_hub(self):
        model = self.load()
        model_repo = HuggingFaceHub(TrainerConfig(), "model_repo")
        model_repo.push_to_hub()
        model_repo.save(model)
        return model_repo

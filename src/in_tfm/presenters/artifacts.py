from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from ..viz import save_image_grid


class ClusterArtifacts(BaseModel):
    """Where a presenter writes one cluster's files, and the url the page links them by.

    The page sits one folder above `cluster_dir`, so urls are relative to that.
    Keeping both here stops presenters from rebuilding `{cluster_dir.name}/...` by hand.
    """

    model_config = ConfigDict(frozen=True)

    cluster_dir: Path

    @property
    def name(self) -> str:
        return self.cluster_dir.name

    def path(self, filename: str) -> Path:
        return self.cluster_dir / filename

    def url(self, filename: str) -> str:
        return f"{self.name}/{filename}"

    def save_grid(
        self,
        images: list[Image.Image],
        filename: str,
        cols: int | None = None,
        pad: int = 4,
        border: int = 0,
    ) -> str:
        save_image_grid(images, self.path(filename), cols=cols, pad=pad, border=border)
        return self.url(filename)

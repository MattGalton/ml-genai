from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class DataSpec:
    data_dim: int
    num_classes: Optional[int] = None
    name: str | None = None
    shape: tuple[int, ...] | None = None

    def _metadata(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

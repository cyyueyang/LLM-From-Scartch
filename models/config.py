from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

@dataclass
class ModelArgs:
    dim: int = 4096
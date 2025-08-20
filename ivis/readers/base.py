# ivis/readers/base.py
from __future__ import annotations
from typing import Protocol, Iterator, Tuple, List, Union
from ..types import VisIData

class Reader(Protocol):
    # metadata (optional but useful)
    def list_ms(self, ms_dir: str) -> List[str]: ...
    def freq_grid(self, ms_dir: str): ...  # -> np.ndarray of Hz

    # I/O
    def read_blocks_I(self, ms_root: str, **kwargs) -> Union[VisIData, List[VisIData]]: ...
    def read_block_I(self, ms_dir: str, **kwargs) -> VisIData: ...
    def iter_channel_slabs(self, ms_dir: str, **kwargs) -> Iterator[Tuple[int, int, VisIData]]: ...

    

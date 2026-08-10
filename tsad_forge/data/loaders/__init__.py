"""데이터셋별 로더. M1에서 SMAP/MSL, SMD, UCR, TSB-AD 로더가 추가된다.

M0에는 BYOD(file) 로더만 있다.
"""

from tsad_forge.data.loaders.file import load_file

__all__ = ["load_file"]

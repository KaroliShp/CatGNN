# Type aliasing
from typing import Tuple
from typing import Callable
from typing import List

import torch


Type_V = int
Type_E = Tuple[int,int]
Type_R = torch.Tensor
Type = List[Type_R]
Type_V_R = Callable[[Type_V], Type_R]
Type_E_R = Callable[[Type_E], Type_R]
Type_V = Callable[[Type_V], Type]
# numpy_compat.py
import numpy as np # type: ignore
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
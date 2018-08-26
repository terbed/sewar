import numpy as np


def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes"
	assert GT.dtype == P.dtype, "Supplied images have different dtypes"
	return GT.astype(np.float64),P.astype(np.float64)
from typing import List, Optional
import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]):
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

def random_weights(dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w

def get_reference_directions(name, *args, **kwargs):
    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
        "energy": RieszEnergyReferenceDirectionFactory,
        "multi-layer": MultiLayerReferenceDirectionFactory,
        "layer-energy": LayerwiseRieszEnergyReferenceDirectionFactory,
        "reduction": ReductionBasedReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from .integrators import BilinearFormIntegrator
from .integrators import LinearFormIntegrator


@dataclass
class BilinearForm:
    """Contatinerclass collecting the individual BilinearFormIntegrators"""

    bilinearforms: List[BilinearFormIntegrator]

    def assemble(self, system_matrix: NDArray[np.floating]):
        for bilinearform in self.bilinearforms:
            bilinearform.assemble(system_matrix)


@dataclass
class LinearForm:
    """Contatinerclass collecting the individual LinearFormIntegrators"""

    linearforms: List[LinearFormIntegrator]

    def assemble(self, system_matrix: NDArray[np.floating]):
        for linearform in self.linearforms:
            linearform.assemble(system_matrix)

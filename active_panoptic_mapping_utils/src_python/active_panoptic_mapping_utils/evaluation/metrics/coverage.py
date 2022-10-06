import numpy as np

from ..constants import COVERAGE_KEY


def coverage(_, pred_vertex_labels):
    # Compute coverage mask
    covered_vertices_mask = pred_vertex_labels != -1

    return {COVERAGE_KEY: np.count_nonzero(covered_vertices_mask)
                            / covered_vertices_mask.size}



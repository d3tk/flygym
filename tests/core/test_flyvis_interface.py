"""RetinaMapper round trip. Skipped when FlyVis is not installed."""

import numpy as np
import pytest

pytest.importorskip("flyvis")

from flygym.vision.flyvis_interface import RetinaMapper


def test_retina_mapper_round_trip():
    values = np.random.default_rng(0).random((2, 2, 721))
    mapper = RetinaMapper()
    assert np.array_equal(
        values, mapper.flyvis_to_flygym(mapper.flygym_to_flyvis(values))
    )
    assert np.array_equal(
        values, mapper.flygym_to_flyvis(mapper.flyvis_to_flygym(values))
    )

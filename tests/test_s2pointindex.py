import pickle

import numpy as np
import pytest

from pys2index import S2PointIndex


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


@pytest.fixture
def index_points(dtype):
    lat, lon = np.meshgrid(np.linspace(-85, 85, 10), np.linspace(-179, 179, 10))
    return np.stack((lat.ravel(), lon.ravel()), axis=1).astype(dtype)


@pytest.fixture
def query_points(index_points, dtype):
    # query point positions shifted randomly
    # - along latitude axis only (easier for testing distances)
    # - small perturbations (< index point spacing)
    npoints = index_points.shape[0]
    perturb_lat = np.random.uniform(-5.0, 5.0, npoints)
    perturn_lon = np.zeros(npoints)

    return (index_points + np.stack((perturb_lat, perturn_lon), axis=1)).astype(dtype)


@pytest.fixture
def distances(index_points, query_points):
    return np.abs(index_points[:, 0] - query_points[:, 0])


def test_s2pointindex_create(index_points):
    # just check that no error is raised at index construction
    index = S2PointIndex(index_points)
    assert isinstance(index, S2PointIndex)


def test_s2pointindex_query(index_points, query_points, distances):
    index = S2PointIndex(index_points)
    dist, pos = index.query(query_points)

    np.testing.assert_array_equal(index_points[pos], index_points)
    np.testing.assert_allclose(dist, distances, atol=1e-6)


def test_s2pointindex_pickle(index_points):
    index = S2PointIndex(index_points)

    expected = index.get_cell_ids()

    pickled = pickle.dumps(index)
    loaded = pickle.loads(pickled)

    np.testing.assert_array_equal(loaded.get_cell_ids(), expected)

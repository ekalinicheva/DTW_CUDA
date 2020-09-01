import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from sklearn.model_selection import cross_val_score, KFold

from tslearn_cuda.not_used.tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn_cuda.not_used.tslearn.svm import TimeSeriesSVC, TimeSeriesSVR
from tslearn_cuda.not_used.tslearn.clustering import GlobalAlignmentKernelKMeans, TimeSeriesKMeans
from tslearn_cuda.not_used.tslearn.utils import to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_variable_length_knn():
    X = to_time_series_dataset([[1, 2, 3, 4],
                                [1, 2, 3],
                                [2, 5, 6, 7, 8, 9],
                                [3, 5, 6, 7, 8]])
    y = [0, 0, 1, 1]
    clf = KNeighborsTimeSeriesClassifier(metric="dtw", n_neighbors=1)
    clf.fit(X, y)
    assert_allclose(clf.predict(X), [0, 0, 1, 1])

    clf = KNeighborsTimeSeriesClassifier(metric="softdtw", n_neighbors=1)
    clf.fit(X, y)
    assert_allclose(clf.predict(X), [0, 0, 1, 1])

def test_variable_length_svm():
    X = to_time_series_dataset([[1, 2, 3, 4],
                                [1, 2, 3],
                                [2, 5, 6, 7, 8, 9],
                                [3, 5, 6, 7, 8]])
    y = [0, 0, 1, 1]
    rng = np.random.RandomState(0)
    clf = TimeSeriesSVC(kernel="gak", random_state=rng)
    clf.fit(X, y)
    assert_allclose(clf.predict(X), [0, 0, 1, 1])

    y_reg = [-1., -1.3, 3.2, 4.1]
    clf = TimeSeriesSVR(kernel="gak")
    clf.fit(X, y_reg)
    assert_array_less(clf.predict(X[:2]), 0.)
    assert_array_less(-clf.predict(X[2:]), 0.)

def test_variable_length_clustering():
    # TODO: here we just check that they can accept variable-length TS, not
    # that they do clever things
    X = to_time_series_dataset([[1, 2, 3, 4],
                                [1, 2, 3],
                                [2, 5, 6, 7, 8, 9],
                                [3, 5, 6, 7, 8]])
    rng = np.random.RandomState(0)

    clf = GlobalAlignmentKernelKMeans(n_clusters=2, random_state=rng)
    clf.fit(X)

    clf = TimeSeriesKMeans(n_clusters=2, metric="dtw", random_state=rng)
    clf.fit(X)

    clf = TimeSeriesKMeans(n_clusters=2, metric="softdtw", random_state=rng)
    clf.fit(X)

def test_variable_cross_val():
    # TODO: here we just check that they can accept variable-length TS, not
    # that they do clever things
    X = to_time_series_dataset([[1, 2, 3, 4],
                                [1, 2, 3],
                                [1, 2, 3, 4],
                                [1, 2, 3],
                                [2, 5, 6, 7, 8, 9],
                                [3, 5, 6, 7, 8],
                                [2, 5, 6, 7, 8, 9],
                                [3, 5, 6, 7, 8]])
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    rng = np.random.RandomState(0)

    cv = KFold(n_splits=2, shuffle=True)
    for estimator in [
        TimeSeriesSVC(kernel="gak", random_state=rng),
        TimeSeriesSVR(kernel="gak"),
        KNeighborsTimeSeriesClassifier(metric="dtw", n_neighbors=1),
        KNeighborsTimeSeriesClassifier(metric="softdtw", n_neighbors=1)
    ]:
        # TODO: cannot test for clustering methods since they don't have a
        # score method yet
        cross_val_score(estimator, X=X, y=y, cv=cv)

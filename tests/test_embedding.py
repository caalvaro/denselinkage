"""Unit tests for ``HashedNGramEmbedder`` (the empty->zero-vector edge case is
covered in ``test_quickstart_end_to_end``)."""

import numpy as np

from denselinkage.embedding import HashedNGramEmbedder


def test_encode_shape_matches_n_features() -> None:
    embedder = HashedNGramEmbedder(n_features=128, ngram=3)
    vectors = embedder.encode(["hello world", "foo"])
    assert vectors.shape == (2, 128)
    assert embedder.embedding_dim == 128


def test_nonempty_rows_are_l2_normalized() -> None:
    vectors = HashedNGramEmbedder().encode(["acme corporation"])
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_encoding_is_deterministic_across_instances() -> None:
    # zlib.crc32 hashing is stable across processes/instances (unlike hash()).
    first = HashedNGramEmbedder(n_features=64).encode(["acme inc"])
    second = HashedNGramEmbedder(n_features=64).encode(["acme inc"])
    assert np.array_equal(first, second)


def test_identical_texts_get_identical_vectors() -> None:
    vectors = HashedNGramEmbedder().encode(["acme", "acme"])
    assert np.array_equal(vectors[0], vectors[1])


def test_model_id_encodes_the_configuration() -> None:
    assert HashedNGramEmbedder(n_features=64, ngram=4).model_id == "hashed-ngram-4-64"

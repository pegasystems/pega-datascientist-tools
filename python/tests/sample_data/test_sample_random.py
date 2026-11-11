import json
import os
import random
import subprocess
import sys

import pytest
from pdstools.sample_data._random import entity_rng, entity_seed


def test_canonical_golden_stream():
    assert entity_seed(42, "model", "first") == (
        31564824182269365280997089106641841006074751843387646939977469325101200009720
    )
    rng = entity_rng(42, "model", "first")
    assert [rng.random() for _ in range(3)] == [
        0.07855024765375263,
        0.5161778644702852,
        0.9108503062419613,
    ]


def test_independent_streams_ignore_catalog_order():
    def samples(names):
        result = {}
        for name in names:
            rng = entity_rng(42, "model", name)
            result[name] = [rng.random() for _ in range(3)]
        return result

    baseline = samples(["first", "second"])
    extended = samples(["new", "second", "first"])
    assert baseline == {name: extended[name] for name in baseline}
    assert baseline["first"] != baseline["second"]


def test_seed_and_identity_change_the_stream():
    assert entity_seed(42, "model", "first") != entity_seed(43, "model", "first")
    assert entity_seed(42, "a/b", "c") != entity_seed(42, "a", "b/c")
    assert entity_seed(42, "a", "b") != entity_seed(42, "b", "a")


def test_does_not_mutate_global_random_state():
    before = random.getstate()
    entity_rng(42, "model", "first").random()
    assert random.getstate() == before


def test_reproducibility_across_process_hash_seeds():
    script = (
        "import json; from pdstools.sample_data._random import entity_rng, entity_seed; "
        "rng = entity_rng(42, 'model', 'first'); "
        "print(json.dumps([entity_seed(42, 'model', 'first'), [rng.random() for _ in range(3)]]))"
    )
    results = [
        json.loads(subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}))
        for seed in ["0", "123"]
    ]
    assert results[0] == results[1]
    assert results[0][0] == entity_seed(42, "model", "first")


@pytest.mark.parametrize("seed", [True, 42.0, "42", None])
def test_reject_noninteger_root_seed(seed):
    with pytest.raises(TypeError, match="integer"):
        entity_seed(seed, "model")


@pytest.mark.parametrize("identity", [(), ("",), ("model", ""), (1,)])
def test_reject_invalid_identity(identity):
    with pytest.raises(ValueError, match="non-empty string"):
        entity_seed(42, *identity)

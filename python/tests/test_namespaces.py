import pathlib
import sys

import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools.utils.namespaces import LazyNamespace, MissingDependenciesException


def test_namespace():
    class NoDependencies(LazyNamespace):
        def test(self): ...

    NoDependencies().test()

    class FulfilledDependencies(LazyNamespace):
        dependencies = ["polars"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    FulfilledDependencies().test()

    class BuiltInDependency(LazyNamespace):
        dependencies = ["colorsys"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    BuiltInDependency().test()

    class MissingDependency(LazyNamespace):
        dependencies = ["fake_dependency"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    with pytest.raises(MissingDependenciesException):
        MissingDependency().test()

    class MissingDependencyWithGroup(LazyNamespace):
        dependencies = ["fake_dependency"]
        dependency_group = "TestGroup"

        def __init__(self):
            super().__init__()

        def test(self): ...

    with pytest.raises(MissingDependenciesException):
        MissingDependencyWithGroup().test()

    with pytest.raises(MissingDependenciesException):
        MissingDependenciesException(["polars"])

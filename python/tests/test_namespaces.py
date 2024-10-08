import pathlib
import sys

import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools.utils.namespaces import LazyNamespace, MissingDependenciesException


def test_no_dependencies():
    class NoDependencies(LazyNamespace):
        def test(self): ...

    NoDependencies().test()


def test_fulfilled_dependencies():
    class FulfilledDependencies(LazyNamespace):
        dependencies = ["polars"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    FulfilledDependencies().test()


def test_builtin_dependency():
    class BuiltInDependency(LazyNamespace):
        dependencies = ["colorsys"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    BuiltInDependency().test()


def test_missing_dependency():
    class MissingDependency(LazyNamespace):
        dependencies = ["fake_dependency"]

        def __init__(self):
            super().__init__()

        def test(self): ...

    with pytest.raises(MissingDependenciesException):
        MissingDependency().test()


def test_missing_dependency_with_group():
    class MissingDependencyWithGroup(LazyNamespace):
        dependencies = ["fake_dependency"]
        dependency_group = "TestGroup"

        def __init__(self):
            super().__init__()

        def test(self): ...

    with pytest.raises(MissingDependenciesException):
        MissingDependencyWithGroup().test()


def test_raising_without_namespace_name():
    with pytest.raises(MissingDependenciesException):
        raise MissingDependenciesException(["polars"])

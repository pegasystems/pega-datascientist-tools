import pytest
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


def test_install_name_mapping_for_yaml():
    """Error message should suggest the pip install name, not the import name."""
    exc = MissingDependenciesException(["yaml"], namespace="Reports", deps_group="explanations")
    assert "pyyaml" in str(exc)
    # The human-readable summary still references the import name the user sees
    assert "yaml" in str(exc)
    assert "explanations" in str(exc)


def test_install_name_mapping_for_polars_hash():
    exc = MissingDependenciesException(["polars_hash"], namespace="Anonymization", deps_group="pega_io")
    assert "polars-hash" in str(exc)


def test_install_hint_is_package_manager_neutral():
    """We mention pip as an example, but shouldn't hardcode `uv` in the hint."""
    exc = MissingDependenciesException(["plotly"], namespace="Plots", deps_group="adm")
    msg = str(exc)
    assert "favorite package manager" in msg
    assert "uv pip install" not in msg

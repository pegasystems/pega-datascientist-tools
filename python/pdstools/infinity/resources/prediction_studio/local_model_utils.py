from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from .base import LocalModel, ModelValidationError

if TYPE_CHECKING:
    from onnx import ModelProto
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PEGA_METADATA = "pegaMetadata"


class OutcomeType(Enum):
    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class Predictor(BaseModel):
    """A single predictor (feature) in an ONNX model.

    **Automatic name derivation** — When *pega_property* is supplied
    (e.g. ``".Customer.Age"``) and *name* is omitted, the predictor
    ``name`` is automatically set to the leaf segment of the property
    path (``"Age"``).  Pega Prediction Studio auto‑maps predictors by
    matching ``name`` against its data‑model properties, so this
    guarantees correct field mapping on upload without any manual work.

    If *name* is provided explicitly it is always used as‑is.

    See Also
    --------
    Metadata.build_predictor_list : Batch‑build predictors from a list
        of Pega property paths or plain feature names.
    """

    name: str | None = Field(default=None, validate_default=True)
    index: int | None = Field(default=None, validate_default=True)
    input_name: str | None = Field(default=None, validate_default=True)
    data_type: str = Field(default="Numeric")
    pega_property: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _derive_name_from_pega_property(cls, data):
        """Auto‑derive ``name`` from the leaf of ``pega_property``.

        Runs before individual field validators so that every
        downstream validator already sees a resolved ``name``.
        """
        if isinstance(data, dict):
            name = data.get("name")
            pega_prop = data.get("pega_property")
            if name is None and pega_prop:
                data["name"] = pega_prop.rsplit(".", 1)[-1]
        return data

    @field_validator("input_name", mode="before")
    def validate_input_name(cls, v):
        if v is None:
            raise ValueError(
                "The 'input_name' for a predictor is absent in the ONNX metadata. "
                "Ensure it is populated for all predictors.",
            )
        return v

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        if v is None:
            raise ValueError(
                "The 'name' for a predictor is absent in the ONNX metadata. "
                "Either provide 'name' explicitly or supply 'pega_property' "
                "so the name can be derived automatically.",
            )
        return v

    @field_validator("index", mode="before")
    def validate_index(cls, v):
        if v is None or v < 1:
            raise ValueError(
                "The 'index' for a predictor is absent/invalid in the ONNX metadata. "
                "Ensure it is populated for all predictors.",
            )
        return v

    @field_validator("data_type", mode="before")
    def validate_data_type(cls, v):
        if v is not None and v not in ("Numeric", "Symbolic"):
            raise ValueError(
                "The 'data_type' for a predictor must be 'Numeric' or 'Symbolic'.",
            )
        return v


class Output(BaseModel):
    possible_values: list[str | int | float] = Field(default_factory=list)
    label_name: str | None = Field(default=None, validate_default=True)
    score_name: str | None = Field(default=None)
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)

    @field_validator("label_name", mode="before")
    def validate_label_name(cls, v):
        if v is None:
            raise ValueError(
                "The 'labelName' is required but missing or empty in the 'output' object of ONNX model metadata. Ensure it is populated",
            )
        return v


class Metadata(BaseModel):
    type: OutcomeType | None = Field(default=None, validate_default=True)
    predictor_list: list[Predictor] = Field(default_factory=list)
    output: Output | None = Field(default=None, validate_default=True)
    modeling_technique: str | None = Field(default=None)
    internal: bool | None = Field(default=None)

    # Enterprise / governance fields (Pega 24+).
    # Python API uses clean names; Pydantic aliases handle the Pega
    # ``py``-prefixed camelCase keys (e.g. ``pyModelVersion``) in JSON.
    file_source: str | None = Field(
        default=None,
        validation_alias=AliasChoices("file_source", "py_file_source"),
        serialization_alias="pyFileSource",
    )
    objective: str | None = Field(
        default=None,
        validation_alias=AliasChoices("objective", "py_objective"),
        serialization_alias="pyObjective",
    )
    rule_set: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rule_set", "py_rule_set"),
        serialization_alias="pyRuleSet",
    )
    rule_set_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rule_set_version", "py_rule_set_version"),
        serialization_alias="pyRuleSetVersion",
    )
    predict_method_uses_name_value_pair: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "predict_method_uses_name_value_pair",
            "py_predict_method_uses_name_value_pair",
        ),
        serialization_alias="pyPredictMethodUsesNameValuePair",
    )
    model_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model_version", "py_model_version"),
        serialization_alias="pyModelVersion",
    )
    created_by: str | None = Field(
        default=None,
        validation_alias=AliasChoices("created_by", "py_created_by"),
        serialization_alias="pyCreatedBy",
    )
    created_date: str | None = Field(
        default=None,
        validation_alias=AliasChoices("created_date", "py_created_date"),
        serialization_alias="pyCreatedDate",
    )
    last_modified_date: str | None = Field(
        default=None,
        validation_alias=AliasChoices("last_modified_date", "py_last_modified_date"),
        serialization_alias="pyLastModifiedDate",
    )
    training_dataset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("training_dataset", "py_training_dataset"),
        serialization_alias="pyTrainingDataset",
    )
    experiment_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("experiment_id", "py_experiment_id"),
        serialization_alias="pyExperimentId",
    )
    parent_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("parent_model_id", "py_parent_model_id"),
        serialization_alias="pyParentModelId",
    )
    baseline_auc: float | None = Field(
        default=None,
        validation_alias=AliasChoices("baseline_auc", "py_baseline_auc"),
        serialization_alias="pyBaselineAUC",
    )
    baseline_accuracy: float | None = Field(
        default=None,
        validation_alias=AliasChoices("baseline_accuracy", "py_baseline_accuracy"),
        serialization_alias="pyBaselineAccuracy",
    )
    performance_threshold: float | None = Field(
        default=None,
        validation_alias=AliasChoices("performance_threshold", "py_performance_threshold"),
        serialization_alias="pyPerformanceThreshold",
    )

    @field_validator("type", mode="before")
    def validate_type(cls, v, values):
        if not values.data.get("internal"):
            supported_types = [output_type.value for output_type in OutcomeType]
            if isinstance(v, OutcomeType):
                v = v.value
            if v is None or v not in supported_types:
                raise ValueError(
                    f"The 'type' in ONNX model metadata has invalid value '{v}'. Ensure it is populated with one of the following supported types: {supported_types}",
                )
        return v

    @field_validator("output", mode="before")
    def validate_output(cls, v, values):
        if not values.data.get("internal") and v is None:
            raise ValueError(
                "The 'output' is required but missing in the ONNX model metadata. Ensure it is populated",
            )
        return v

    def to_json(self) -> str:
        return json.dumps(self, cls=self._ONNXMetadataEncoder, separators=(", ", ": "))

    @classmethod
    def from_json(cls, json_str: str) -> Metadata:
        data = json.loads(json_str)
        data_snake_case = cls._convert_keys(data, cls._to_snake_case)
        return cls.model_validate(data_snake_case)

    @staticmethod
    def build_predictor_list(
        features: list[str],
        input_name: str = "features",
        *,
        data_types: list[str] | dict[str, str] | None = None,
    ) -> list[Predictor]:
        """Build a predictor list from feature names or Pega property paths.

        This is the **recommended** one‑liner for constructing
        predictors.  Each entry in *features* can be:

        * A **Pega property path** starting with ``"."``
          (e.g. ``".Customer.Age"``).  The leaf segment becomes the
          predictor ``name`` (``"Age"``) and the full path is stored as
          ``pega_property`` so Pega Prediction Studio auto‑maps the
          field on upload.
        * A **plain feature name** (e.g. ``"Age"``).  Used as‑is for
          ``name``; ``pega_property`` is left unset.

        Indices are assigned automatically (1‑based) in the order the
        features appear.

        Parameters
        ----------
        features
            Ordered list of feature identifiers.  The order **must**
            match the column order in the ONNX input tensor.  Accepts
            any mix of plain names and Pega property paths.
        input_name
            Name of the ONNX input node.  Default ``"features"``.
        data_types
            Optional type annotations for features.  Accepts either:

            * A **list** of ``"Numeric"`` / ``"Symbolic"`` values,
              one per feature (must match *features* length).
            * A **dict** mapping feature *names* (the leaf segment for
              Pega paths) to ``"Numeric"`` or ``"Symbolic"``.  Only the
              features present in the dict are overridden; the rest
              default to ``"Numeric"``.

            When ``None`` every feature defaults to ``"Numeric"``.

        Returns
        -------
        list[Predictor]

        Examples
        --------
        >>> # Using Pega property paths (recommended for auto-mapping):
        >>> Metadata.build_predictor_list(
        ...     [".Customer.Age", ".Customer.Tenure", ".Customer.MonthlyCharges"],
        ... )  # doctest: +SKIP

        >>> # Using plain feature names:
        >>> Metadata.build_predictor_list(["Age", "Tenure", "MonthlyCharges"])
        ... # doctest: +SKIP

        >>> # Sparse overrides via dict:
        >>> Metadata.build_predictor_list(
        ...     [".Customer.Age", ".Customer.ContractType"],
        ...     data_types={"ContractType": "Symbolic"},
        ... )  # doctest: +SKIP

        """
        if isinstance(data_types, list) and len(data_types) != len(features):
            raise ValueError(f"Length of data_types ({len(data_types)}) must match features ({len(features)}).")

        dt_dict: dict[str, str] | None = None
        if isinstance(data_types, dict):
            dt_dict = data_types
        elif isinstance(data_types, list):
            dt_dict = None  # handled via index below

        predictors: list[Predictor] = []
        for idx, feat in enumerate(features, start=1):
            is_pega_path = feat.startswith(".")
            leaf = feat.rsplit(".", maxsplit=1)[-1] if is_pega_path else feat
            kwargs: dict = {
                "name": None if is_pega_path else feat,
                "index": idx,
                "input_name": input_name,
                "pega_property": feat if is_pega_path else None,
            }
            if isinstance(data_types, list):
                kwargs["data_type"] = data_types[idx - 1]
            elif dt_dict is not None and leaf in dt_dict:
                kwargs["data_type"] = dt_dict[leaf]
            predictors.append(Predictor(**kwargs))
        return predictors

    @staticmethod
    def _convert_keys(data: dict, conversion_func) -> dict:
        if isinstance(data, dict):
            return {conversion_func(k): Metadata._convert_keys(v, conversion_func) for k, v in data.items()}
        if isinstance(data, list):
            return [Metadata._convert_keys(i, conversion_func) for i in data]
        return data

    @staticmethod
    def _to_snake_case(string: str) -> str:
        # Handle transitions like "baselineAUC" -> "baseline_AUC" -> "baseline_auc"
        # and "predictMethodUsesNameValuePair" correctly.
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return s.lower()

    @staticmethod
    def _to_camel_case(string: str) -> str:
        parts = string.split("_")
        result = parts[0]
        for word in parts[1:]:
            if word.lower() == "auc":
                result += word.upper()
            else:
                result += word.capitalize()
        return result

    class _ONNXMetadataEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Enum):
                return o.value
            if isinstance(o, BaseModel):
                dumped = Metadata._strip_empty(o.model_dump(by_alias=True, exclude_none=True))
                return Metadata._convert_keys(dumped, Metadata._to_camel_case)
            return super().default(o)

    @staticmethod
    def _strip_empty(data):
        """Recursively remove keys whose value is ``[]``."""
        if isinstance(data, dict):
            return {k: Metadata._strip_empty(v) for k, v in data.items() if v != []}
        if isinstance(data, list):
            return [Metadata._strip_empty(i) for i in data]
        return data


class ONNXModelCreationError(Exception):
    """Exception for errors during ONNX conversion and save."""


class ONNXModelValidationError(ModelValidationError):
    """Exception for errors during ONNX validation."""


class PMMLModel(LocalModel):
    file_path: str

    def __init__(self, file_path: str):
        super().__init__(file_path=file_path)  # type: ignore[call-arg]

    def get_file_path(self) -> str:
        return self.file_path


class H2OModel(LocalModel):
    file_path: str

    def __init__(self, file_path: str):
        super().__init__(file_path=file_path)  # type: ignore[call-arg]

    def get_file_path(self) -> str:
        return self.file_path


class ONNXModel(LocalModel):
    _model: ModelProto

    def __init__(self, model: ModelProto):
        super().__init__()
        from onnx import ModelProto

        if isinstance(model, ModelProto):
            self._model = model
            return

        raise ONNXModelCreationError("Model must be a onnx ModelProto object")

    @classmethod
    def from_onnx_proto(cls, model: ModelProto) -> ONNXModel:
        """Creates an ONNXModel object.

        Parameters
        ----------
        model : ModelProto
            An onnx ModelProto object

        Returns
        -------
        ONNXModel
            An instance of the ONNXModel class.

        Raises
        ------
        ImportError
            If the optional dependencies for ONNX Conversion are not installed.
        ONNXModelCreationError
            If the ONNXModel object cannot be created.

        """
        return cls(model=model)

    @classmethod
    def from_sklearn_pipeline(cls, model: Pipeline, initial_types: list) -> ONNXModel:
        """Creates an ONNXModel object.

        Parameters
        ----------
        model : Pipeline
            A sklearn Pipeline object

        initial_types : list
            A list of initial types for the model's input variables if the model is a Sklearn Pipeline object.

        Returns
        -------
        ONNXModel
            An instance of the ONNXModel class.

        Raises
        ------
        ImportError
            If the optional dependencies for ONNX Conversion are not installed.
        ONNXModelCreationError
            If the ONNXModel object cannot be created.

        """
        from skl2onnx import convert_sklearn
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            model = convert_sklearn(model, initial_types=initial_types)
            return cls(model=model)

        raise ONNXModelCreationError("Model must be a sklearn Pipeline object.")

    @classmethod
    def from_pytorch(
        cls,
        model,
        dummy_input,
        *,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        opset_version: int = 17,
        fixed_batch_size: bool = True,
    ) -> ONNXModel:  # pragma: no cover
        """Create an ONNXModel from a PyTorch ``nn.Module``.

        The model is exported via ``torch.onnx.export`` with static shapes.
        When *fixed_batch_size* is ``True`` (the default), any remaining
        dynamic dimensions are replaced with a batch size of 1 so that the
        resulting ONNX file is accepted by Pega Prediction Studio.

        Parameters
        ----------
        model
            A PyTorch ``nn.Module`` (already in eval mode is recommended).
        dummy_input
            Example input tensor(s) matching the model's forward signature.
        input_names
            Optional list of ONNX input node names.  Defaults to ``["input"]``.
        output_names
            Optional list of ONNX output node names.  Defaults to ``["output"]``.
        opset_version
            ONNX opset version.  Default ``17``.
        fixed_batch_size
            Replace dynamic dimensions with batch size 1.

        Returns
        -------
        ONNXModel

        Raises
        ------
        ONNXModelCreationError
            If PyTorch is not installed or the export fails.

        """
        try:
            import torch
        except ImportError:
            raise ONNXModelCreationError(
                "PyTorch is required for from_pytorch(). Install it with: uv pip install torch"
            )

        import io as _io

        import onnx

        model.eval()
        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        buf = _io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                buf,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=None,
                opset_version=opset_version,
                do_constant_folding=True,
            )

        buf.seek(0)
        proto = onnx.load_model_from_string(buf.read())

        if fixed_batch_size:
            cls._fix_dynamic_shapes(proto, batch_size=1)

        logger.info("PyTorch model exported to ONNX (opset %s).", opset_version)
        return cls(model=proto)

    # ------------------------------------------------------------------
    # Shape fixing
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_dynamic_shapes(proto: ModelProto, batch_size: int = 1) -> None:  # pragma: no cover
        """Replace dynamic (symbolic) dimensions with a fixed *batch_size*.

        Pega Prediction Studio rejects ONNX models whose input or output
        nodes contain symbolic dimension parameters (e.g. ``"batch"``).
        This helper iterates every dimension on every input/output and
        replaces symbolic dims with the given integer value.

        Parameters
        ----------
        proto
            An ``onnx.ModelProto`` — **modified in place**.
        batch_size
            The integer value to substitute.  Default ``1``.

        """
        for node in list(proto.graph.input) + list(proto.graph.output):
            if node.type.HasField("tensor_type"):
                for dim in node.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dim.ClearField("dim_param")
                        dim.dim_value = batch_size
        logger.debug("Fixed dynamic shapes to batch_size=%d.", batch_size)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_metadata(self) -> Metadata | None:
        """Return the embedded ``Metadata`` or ``None`` if absent."""
        for prop in self._model.metadata_props:
            if prop.key == PEGA_METADATA:
                return Metadata.from_json(prop.value)
        return None

    def add_metadata(self, metadata: Metadata) -> ONNXModel:
        """Adds metadata to the ONNX model.

        Parameters
        ----------
        metadata : Meta
            The metadata to be added.

        Returns
        -------
        ONNXModel
            The ONNXModel object with the added metadata.

        Raises
        ------
        ImportError
            If the optional dependencies for ONNX Metadata addition are not installed.

        """
        if PEGA_METADATA in self._model.metadata_props:
            self._model.metadata_props.remove(PEGA_METADATA)
        self._model.metadata_props.add(key=PEGA_METADATA, value=metadata.to_json())
        return self

    def validate(self) -> bool:  # type: ignore[override]
        """Validates an ONNX model.

        Raises
        ------
            ImportError: If the optional dependencies for ONNX Validation are not installed.
            ONNXModelValidationError: If the model is invalid or if the validation process fails.

        """
        session = None
        from io import StringIO

        import onnx
        from onnxruntime import InferenceSession

        error_stream = StringIO()
        onnx.checker.check_model(self._model)
        try:
            session = InferenceSession(self._model.SerializeToString())
        except Exception as e:
            raise ONNXModelValidationError(
                f"Unable to create inference session: {e!s}",
            )
        metadata = session.get_modelmeta().custom_metadata_map

        if PEGA_METADATA not in metadata:
            raise ONNXModelValidationError(
                "The 'pegaMetadata' key is absent in the ONNX model's metadata. "
                "Ensure the metadata includes a 'pegaMetadata' key "
                "with a JSON object value as described in pega documentation."
                "Adhering to this structure is crucial for accurate model execution.",
            )

        metadata_str = metadata[PEGA_METADATA]
        metadata = Metadata.from_json(metadata_str)

        valid_input = self.__check_for_valid_input_node_structure(
            error_stream,
            session,
            metadata,
        )
        valid_output = metadata.internal or self.__check_for_valid_output_node_structure(
            error_stream,
            session,
            metadata,
        )
        if not (valid_input and valid_output):
            raise ONNXModelValidationError(
                f"Validation failed: {error_stream.getvalue()}",
            )
        del session
        logger.info(
            "ONNX model input/output structure and metadata validation is successful.",
        )
        return True

    def run(self, test_data: dict):
        """Run the prediction using the provided test data.

        Parameters
        ----------
        test_data : dict
            The test data to be used for prediction. It is a dictionary where each key is a column name from the dataset, and each value is a NumPy array representing the column data as a vector. For example:

            {
                'column1': array([[value1], [value2], [value3]]),
                'column2': array([[value4], [value5], [value6]]),
                'column3': array([[value7], [value8], [value9]])
            }

        Returns
        -------
        Any
            The prediction result.

        """
        session = None
        from onnxruntime import InferenceSession

        session = InferenceSession(self._model.SerializeToString())
        result = session.run(None, test_data)
        del session
        return result

    def save(self, onnx_file_path: str):
        """Saves the ONNX model to the specified file path.

        Parameters
        ----------
        onnx_file_path : str
            The file path where the ONNX model should be saved.

        Raises
        ------
            ImportError: If the optional dependencies for ONNX Conversion are not installed.

        """
        import onnx

        onnx.save_model(self._model, onnx_file_path)

    def __check_for_valid_input_node_structure(
        self,
        error_stream,
        session,
        metadata,
    ) -> bool:
        """Checks if the output node structure of the ONNX model is valid.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        session : onnxruntime.InferenceSession
            The ONNX runtime session containing the model.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if the output node structure is valid, False otherwise.

        """
        model_input_info = {i.name: i for i in session.get_inputs()}
        return (
            self.__validate_input_nodes(error_stream, model_input_info, metadata)
            and self.__validate_predictor_mappings(
                error_stream,
                model_input_info,
                metadata,
            )
            and self.__validate_predictor_index_mappings(error_stream, metadata)
        )

    def __validate_input_nodes(self, error_stream, model_input_info, metadata) -> bool:
        """Validates the input nodes of the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_input_info : dict
            A dictionary containing information about the model's input nodes.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if the input nodes are valid, False otherwise.

        """
        return (
            self.__validate_input_node_shapes(error_stream, model_input_info)
            and self.__validate_input_node_dimensions(error_stream, model_input_info)
            and self.__validate_input_node_sizes(
                error_stream,
                model_input_info,
                metadata,
            )
        )

    def __check_for_valid_output_node_structure(
        self,
        error_stream,
        session,
        metadata,
    ) -> bool:
        """Checks if the output node structure of the ONNX model is valid.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        session : onnxruntime.InferenceSession
            The ONNX runtime session containing the model.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if the output node structure is valid, False otherwise.

        """
        model_output_info = {o.name: o for o in session.get_outputs()}
        return self.__validate_label_output_node_exist(
            error_stream,
            model_output_info,
            metadata,
        ) and self.__validate_tensor_output_node_structure(
            error_stream,
            metadata.output.label_name,
            model_output_info[metadata.output.label_name],
        )

    def __validate_tensor_output_node_structure(
        self,
        error_stream,
        node_name,
        value_info,
    ) -> bool:
        """Validates the tensor output node structure of the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        node_name : str
            The name of the output node.
        value_info : Any
            The value information of the output node.

        Returns
        -------
        bool
            True if the output node is of type Tensor, False otherwise.

        """
        if "tensor" not in value_info.type.lower():
            error_stream.write(
                f"Expected the ONNX model's output node '{node_name}' to be of data type Tensor, but found a Map or Sequence instead.",
            )
            return False

        tensor_length = len(value_info.shape)
        is_valid_dimension = tensor_length in [1, 2]
        if not is_valid_dimension:
            error_stream.write(
                f"The shape of the ONNX model's output node '{node_name}' does not comply to the expected 1 or 2 dimensions.",
            )
        return is_valid_dimension

    def __validate_label_output_node_exist(
        self,
        error_stream,
        model_output_info,
        metadata,
    ) -> bool:
        """Validates the existence of the label output node in the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_output_info : dict
            A dictionary containing information about the model's output nodes.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if the label output node exists, False otherwise.

        """
        if metadata.output.label_name not in model_output_info:
            error_stream.write(
                f"The ONNX model does not contain the expected output node for labels, identified by: {metadata.output.label_name}. "
                "Ensure the model's metadata correctly maps the label output node.",
            )
            return False
        return True

    def __validate_input_node_dimensions(self, error_stream, model_input_info) -> bool:
        """Validates the dimensions of the input nodes in the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_input_info : dict
            A dictionary containing information about the model's input nodes.

        Returns
        -------
        bool
            True if all input nodes have valid dimensions, False otherwise.

        """
        predictor_with_dynamic_size = [
            f"{name}({value_info.shape})"
            for name, value_info in model_input_info.items()
            if any(dim is not None and not isinstance(dim, int) for dim in value_info.shape)
        ]
        if predictor_with_dynamic_size:
            error_stream.write(
                "The ONNX model's input nodes are dynamically sized, but fixed array sizes were expected for: "
                + ", ".join(predictor_with_dynamic_size)
                + ". Ensure all input nodes have predefined fixed sizes.",
            )
            return False
        return True

    def __validate_input_node_shapes(self, error_stream, model_input_info) -> bool:
        """Validates the dimensions of the input nodes in the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_input_info : dict
            A dictionary containing information about the model's input nodes.

        Returns
        -------
        bool
            True if all input nodes have valid dimensions, False otherwise.

        """
        predictor_with_invalid_shape = [
            f"{name}({info.shape})" for name, info in model_input_info.items() if len(info.shape) != 2
        ]
        if predictor_with_invalid_shape:
            error_stream.write(
                "The ONNX model's input nodes does not comply to the expected 2-dimensional input shape for: "
                + ", ".join(predictor_with_invalid_shape)
                + ". Verify that all input nodes are correctly shaped with 2 dimensions.",
            )
            return False
        return True

    def __validate_predictor_mappings(
        self,
        error_stream,
        model_input_info,
        metadata,
    ) -> bool:
        """Validates the predictor mappings in the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_input_info : dict
            A dictionary containing information about the model's input nodes.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if all predictor mappings are valid, False otherwise.

        """
        missing_predictors = ""

        if not metadata.predictor_list:
            any_array_predictor_input = any(info.shape[1] != 1 for info in model_input_info.values())
            if any_array_predictor_input:
                missing_predictors = self.__get_missing_predictors(model_input_info, [])
        else:
            predictor_input_names = list(
                set(p.input_name for p in metadata.predictor_list),
            )
            missing_predictors = self.__get_missing_predictors(
                model_input_info,
                predictor_input_names,
            )

        if missing_predictors:
            error_stream.write(
                "Predictor mappings for the following input nodes are missing: "
                + missing_predictors
                + ". Ensure all required input nodes are correctly mapped in the ONNX model's metadata.",
            )
            return False

        return True

    def __validate_predictor_index_mappings(self, error_stream, metadata) -> bool:
        """Validates the predictor index mappings in the ONNX model's metadata.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        metadata : Meta
            The metadata associated with the model.

        Returns
        -------
        bool
            True if the predictor index mappings are valid, False otherwise.

        """
        if metadata.predictor_list is not None:
            predictor_map = self.__create_predictor_map(metadata)
            duplicate_index_mapped_input = [
                input_name
                for input_name, predictors in predictor_map.items()
                if len(set(p.index for p in predictors)) != len(predictors)
            ]
            if duplicate_index_mapped_input:
                error_stream.write(
                    "Detected a mismatch between the ONNX input node size and the number of predictors specified in the metadata for: "
                    + ", ".join(duplicate_index_mapped_input)
                    + ". Ensure the size of each input node matches the corresponding number of predictors defined in the metadata.",
                )
                return False
        return True

    def __create_predictor_map(self, metadata):
        """Creates a mapping of input names to their corresponding predictors.

        Parameters
        ----------
        metadata : Meta
            The metadata associated with the model, containing the predictor list.

        Returns
        -------
        dict
            A dictionary where the keys are input names and the values are lists of predictors.

        """
        predictor_map = {}
        for predictor in metadata.predictor_list:
            input_name = predictor.input_name
            if input_name not in predictor_map:
                predictor_map[input_name] = []
            predictor_map[input_name].append(predictor)
        return predictor_map

    def __validate_input_node_sizes(
        self,
        error_stream,
        model_input_info,
        metadata,
    ) -> bool:
        """Validates the sizes of the input nodes in the ONNX model.

        Parameters
        ----------
        error_stream : Any
            The stream to which error messages are written.
        model_input_info : dict
            A dictionary containing information about the model's input nodes.
        metadata : Meta
            The metadata associated with the model, containing the predictor list.

        Returns
        -------
        bool
            True if all input nodes have valid sizes, False otherwise.

        """
        if metadata.predictor_list is not None:
            model_input_info_map = {name: info.shape[1] for name, info in model_input_info.items()}
            predictor_map = self.__create_predictor_map(metadata)
            predictor_with_invalid_size = [
                f"{name}{model_input_info[name].shape}"
                for name, predictors in predictor_map.items()
                if name in model_input_info_map
                and (
                    len(predictors) != model_input_info_map[name]
                    or any(
                        p.index < 1 or p.index > model_input_info_map[name] if p.index is not None else True
                        for p in predictors
                    )
                )
            ]
            if predictor_with_invalid_size:
                error_stream.write(
                    "Discrepancy detected in ONNX model metadata predictor index mappings versus model's input dimensions for: "
                    + ", ".join(predictor_with_invalid_size)
                    + ". Verify and align the predictor and index mappings with the model's input dimensions.",
                )
                return False
        return True

    def __get_missing_predictors(self, model_input_info, predictor_input_names) -> str:
        """Identifies the missing predictors in the ONNX model's input nodes.

        Parameters
        ----------
        model_input_info : dict
            A dictionary containing information about the model's input nodes.
        predictor_input_names : list
            A list of predictor input names.

        Returns
        -------
        str
            A comma-separated string of missing predictor names.

        """
        return ", ".join(name for name in model_input_info.keys() if name not in predictor_input_names)

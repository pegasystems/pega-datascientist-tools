import polars as pl


class pyValueFinder:
    pyDirection = pl.Categorical
    pySubjectType = pl.Categorical
    ModelPositives = pl.UInt32
    pyGroup = pl.Categorical
    pyPropensity = pl.Float64
    FinalPropensity = pl.Float64
    pyStage = pl.Enum(
        [
            "Eligibility",
            "Applicability",
            "Suitability",
            "Arbitration",
        ]
    )
    pxRank = pl.UInt16
    pxPriority = pl.Float64
    pyModelPropensity = pl.Float64
    pyChannel = pl.Categorical
    Value = pl.Float64
    pyName = pl.Utf8
    StartingEvidence = pl.UInt32
    pySubjectID = pl.Utf8
    DecisionTime = pl.Datetime
    pyTreatment = pl.Utf8
    pyIssue = pl.Categorical

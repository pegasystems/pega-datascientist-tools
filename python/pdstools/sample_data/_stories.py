"""Canonical timed stories shared by all future sample views."""

from .schema import Story

STORIES = (
    Story(
        name="email_feed_dip",
        channel="Email",
        technique="NaiveBayes",
        start_day=112,
        duration_days=14,
        recovery_days=21,
        ctr_multiplier=0.65,
        positive_only=False,
        rationale="A two-week feed-quality problem in weeks 17-18, followed by recovery.",
        serves=("performance_trend", "success_rate", "prediction_lift"),
    ),
    Story(
        name="email_positive_only",
        channel="Email",
        technique="NaiveBayes",
        start_day=14,
        duration_days=7,
        recovery_days=7,
        ctr_multiplier=1,
        positive_only=True,
        rationale="Delayed negative feedback in week 3 produces a 100% measured weekly success rate.",
        serves=("success_rate_anomalies", "response_loop_guidance"),
    ),
    Story(
        name="mobile_positive_only",
        channel="Mobile",
        technique="NaiveBayes",
        start_day=49,
        duration_days=7,
        recovery_days=7,
        ctr_multiplier=1,
        positive_only=True,
        rationale="Week 8 exercises the response-loop marker in the default Mobile NB Model Report.",
        serves=("success_rate_anomalies", "model_report"),
    ),
    Story(
        name="web_positive_only",
        channel="Web",
        technique="GradientBoost",
        start_day=133,
        duration_days=7,
        recovery_days=7,
        ctr_multiplier=1,
        positive_only=True,
        rationale="One week-20 event spans both Web AGB configurations without adding extra channel-level markers.",
        serves=("success_rate_anomalies", "specialized_models"),
    ),
)

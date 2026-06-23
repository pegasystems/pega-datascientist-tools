"""Quality Report page — zero-functionality presentation layer.

All computation is delegated to ``TopicDataQuality.health.*`` and
``TopicDataQuality.plot.*``.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.data_quality.dq_streamlit_utils import ensure_topic_dq
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Quality Report · Topic Data Quality")

dq = ensure_topic_dq()

health = dq.health.calculate_health_score()
tq = dq.health.analyze_text_quality()
dup_info = dq.health.find_duplicates()
duplicates = dup_info["duplicates"]
cross_topic = dup_info["cross_topic"]
duplicate_pct = dup_info["duplicate_pct"]
adequacy_df = dq.health.check_sample_adequacy()
sim_df = dq.compute.topic_similarity()
high_sim_pairs = dq.high_similarity_pairs()
tightness_df = dq.health.calculate_cluster_tightness()
similarity_threshold = dq.similarity_threshold

# ===========================
# HEALTH SCORE
# ===========================
color_map = {
    "GREEN": "#28a745",
    "YELLOW": "#ffc107",
    "ORANGE": "#fd7e14",
    "RED": "#dc3545",
}

st.markdown(
    f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;">
    <h1 style="margin: 0; font-size: 3em; color: {color_map[health["color"]]};">{health["score"]}/100</h1>
    <h3 style="margin: 10px 0;">Dataset Health Status: {health["status"]}</h3>
    <p style="margin: 0; opacity: 0.9;">
        {"Your dataset is ready for model training!" if health["score"] >= 70 else "Address the issues below before training"}
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if health["reasons"]:
    with st.expander("How was this score calculated?", expanded=False):
        st.markdown("**Score breakdown:**")
        for reason in health["reasons"]:
            st.write(f"• {reason}")

# ===========================
# QUICK STATS
# ===========================
st.markdown("---")
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", f"{dq.total_samples:,}")
with col2:
    st.metric("Unique Topics", dq.num_topics)
with col3:
    st.metric("Avg Text Length", f"{tq['avg_words']:.0f} words")
with col4:
    st.metric(
        "Balance Ratio",
        f"{dq.imbalance_ratio:.1f}x",
        delta="High" if dq.imbalance_ratio > 3 else "Good",
        delta_color="inverse" if dq.imbalance_ratio > 3 else "normal",
    )

# ===========================
# SAMPLE ADEQUACY
# ===========================
st.markdown("---")
st.subheader("Sample Size Analysis")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(adequacy_df.to_pandas(), height=min(400, (adequacy_df.height + 1) * 35 + 3))
with col2:
    critical_count = adequacy_df.filter(pl.col("Status").str.contains("CRITICAL")).height
    low_count = adequacy_df.filter(pl.col("Status").str.contains("LOW")).height
    good_count = adequacy_df.filter(pl.col("Status").str.contains("GOOD|ACCEPTABLE")).height
    st.markdown("### Status Summary")
    st.metric("Critical Issues", critical_count)
    st.metric("Low Sample Topics", low_count)
    st.metric("Adequate Topics", good_count)

# ===========================
# TOPIC DISTRIBUTION
# ===========================
st.markdown("---")
st.subheader("Topic Distribution")

fig_bar = dq.plot.topic_distribution()
st.plotly_chart(fig_bar)

if dq.imbalance_ratio > 5:
    st.error(
        f"**Severe Imbalance Detected!** Largest topic has **{dq.imbalance_ratio:.1f}x** "
        "more samples than the smallest."
    )
elif dq.imbalance_ratio > 3:
    st.warning(f"**Moderate Imbalance:** Largest topic has **{dq.imbalance_ratio:.1f}x** more samples than smallest.")
else:
    st.success("**Well-balanced dataset!** All topics have similar representation.")

# ===========================
# TEXT QUALITY
# ===========================
st.markdown("---")
st.subheader("Text Quality Analysis")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Median Length", f"{tq['median_words']:.0f} words")
with col2:
    st.metric("Very Short Texts", f"{tq['short_texts_pct']:.1f}%")
with col3:
    st.metric("Duplicate Texts", f"{duplicate_pct:.1f}%")
with col4:
    st.metric("Very Long Texts", f"{tq['very_long_pct']:.1f}%")

if cross_topic.height > 0:
    st.error(f"**[CRITICAL]** {cross_topic.height} texts appear in multiple topics — fix labelling errors immediately.")
    with st.expander("View cross-topic duplicates"):
        cross_texts = cross_topic.get_column(dq.text_col).to_list()
        for text in cross_texts[:10]:
            st.markdown(f"**Text:** {text[:100]}…")
            dup_topics = duplicates.filter(pl.col(dq.text_col) == text).get_column(dq.topic_col).unique().to_list()
            st.write(f"Appears in topics: {', '.join(map(str, dup_topics))}")
            st.markdown("---")

# ===========================
# TOPIC SEPARATION MAP
# ===========================
st.markdown("---")
st.subheader("Topic Separation Map")

fig_umap = dq.plot.umap_2d()
st.plotly_chart(fig_umap)

st.info(
    """
**How to read this chart:** Each dot is one text sample, coloured by its topic label.
The algorithm places semantically similar texts close together.

- **Tight, separated clusters** → topics are well-defined; the model will classify them easily.
- **Overlapping clusters** → those topics share similar language; the model will confuse them.
- **Scattered dots of one colour** → that topic is poorly defined or has inconsistent labelling.
- **A single dot far from its cluster** → potential mislabelling (see "Potential Labelling Errors" below).
"""
)

with st.expander("Topic Cluster Quality Scores", expanded=True):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(tightness_df.to_pandas())
    with col2:
        avg_tightness = tightness_df.get_column("Tightness Score").mean() if tightness_df.height > 0 else 0
        st.metric("Average Tightness", f"{avg_tightness:.1f}/100")

# ===========================
# TOPIC SIMILARITY
# ===========================
st.markdown("---")
st.subheader("Topic Overlap Analysis")

tab1, tab2 = st.tabs(["Heatmap", "Detailed Matrix"])

with tab1:
    fig_heat = dq.plot.similarity_heatmap()
    st.plotly_chart(fig_heat)

with tab2:
    st.dataframe(sim_df.to_pandas().set_index("topic"))

if high_sim_pairs:
    st.error(
        f"**Problem detected:** {len(high_sim_pairs)} topic pair(s) exceed {similarity_threshold} similarity threshold."
    )
    keyword_analysis = dq.health.keyword_overlap_analysis()
    for kw in keyword_analysis:
        with st.expander(f"'{kw['topic_a']}' vs '{kw['topic_b']}' ({kw['keyword_overlap_pct']}% overlap)"):
            st.markdown(f"**Shared Keywords:** `{', '.join(kw['shared_keywords'])}`")
            st.markdown(f"**Unique to {kw['topic_a']}:** `{', '.join(kw['unique_to_a'])}`")
            st.markdown(f"**Unique to {kw['topic_b']}:** `{', '.join(kw['unique_to_b'])}`")
else:
    st.success(f"**Excellent topic separation!** No topic pairs exceed {similarity_threshold} similarity threshold.")

# ===========================
# CONFUSED SAMPLES
# ===========================
st.markdown("---")
st.subheader("Samples at Risk of Confusion")

confused_df = dq.health.find_confused_samples(top_n=15)
if confused_df.height > 0:
    st.dataframe(confused_df.to_pandas(), height=400)
    csv_confused = confused_df.write_csv()
    st.download_button(
        "Download Confused Samples",
        data=csv_confused,
        file_name="confused_samples.csv",
        mime="text/csv",
    )
else:
    st.success("No high-risk confused samples detected!")

# ===========================
# OUTLIERS
# ===========================
st.markdown("---")
st.subheader("Potential Labelling Errors")

outliers_df = dq.health.find_outliers()
if outliers_df.height > 0:
    st.dataframe(
        outliers_df.sort("distance", descending=True).head(20).to_pandas(),
    )
else:
    st.success("No significant outliers detected – good label consistency!")

# ===========================
# CLEANLAB ANALYSIS
# ===========================
st.markdown("---")
st.subheader("Cleanlab Data Quality Audit")

st.markdown(
    "**[Cleanlab](https://docs.cleanlab.ai/)** uses confident learning "
    "to automatically detect label issues, outliers, and near-duplicates "
    "in your dataset — powered by cross-validated model predictions."
)

cv_warning = dq.compute.cross_validated_checks_warning()
if cv_warning is not None:
    st.warning(cv_warning)
else:
    cleanlab_results = dq.compute.cleanlab_audit()

    # Topic learnability scorecard
    fig_learn = dq.plot.learnability_scorecard()
    st.plotly_chart(fig_learn)

    st.info(
        """
    **How to read this chart:** Each bar shows how well a classifier can learn to
    recognise that topic (F1 score from 5-fold cross-validation on sentence embeddings).

    - **Green (>0.8)** → topic is easily learnable; distinct language patterns.
    - **Orange (0.5–0.8)** → needs attention; may overlap with other topics or lack samples.
    - **Red (<0.5)** → problematic; the model struggles to distinguish this topic. Consider merging, relabelling, or adding more examples.
    """
    )

    # Issue summary chart
    fig_summary = dq.plot.cleanlab_issue_summary()
    st.plotly_chart(fig_summary)

    # Label quality distribution
    fig_lq = dq.plot.label_quality_histogram()
    st.plotly_chart(fig_lq)

    st.info(
        """
    **How to read the histogram:** Each bar shows how many samples received a given
    label quality score (0–1). Samples in **red** (score < threshold) are flagged
    as likely mislabeled by Cleanlab's confident learning algorithm.

    - **Scores near 1.0** → label is very likely correct.
    - **Scores near 0.0** → label is almost certainly wrong; review these first.
    """
    )

    # Top mislabeled samples table
    label_issues = cleanlab_results["label_issues"]
    flagged = label_issues[label_issues["is_label_issue"]].sort_values("label_score")

    if len(flagged) > 0:
        st.markdown(f"**{len(flagged)} samples flagged as likely mislabeled:**")
        texts = dq.df.get_column(dq.text_col).to_list()
        top = flagged.head(20)
        display_df = pl.DataFrame(
            {
                "Text": [texts[i][:120] for i in top.index],
                "Given Label": top["given_label"].tolist(),
                "Suggested Label": top["predicted_label"].tolist(),
                "Label Score": [round(s, 4) for s in top["label_score"].tolist()],
            }
        )
        st.dataframe(display_df.to_pandas(), height=min(500, (min(20, len(flagged)) + 1) * 35 + 3))

        csv_flagged = display_df.write_csv()
        st.download_button(
            "Download Mislabeled Samples (CSV)",
            data=csv_flagged,
            file_name="cleanlab_label_issues.csv",
            mime="text/csv",
        )
    else:
        st.success("No label issues detected by Cleanlab – labels look clean!")

# ===========================
# RECOMMENDATIONS
# ===========================
st.markdown("---")
st.subheader("Action Plan & Recommendations")

recs = dq.health.generate_recommendations()

for priority_label, key in [
    ("🔴 HIGH PRIORITY – Must Fix", "high"),
    ("🟡 MEDIUM PRIORITY – Recommended", "medium"),
    ("🟢 LOW PRIORITY – Nice to Have", "low"),
]:
    items = recs[key]
    if not items:
        continue
    st.markdown(f"### {priority_label}")
    for item in items:
        st.markdown(
            f"**{item['Action']}**\n- **Why:** {item['Why']}\n- **How:** {item['How']}\n- **Impact:** {item['Impact']}"
        )
        st.markdown("---")

if not any(recs.values()):
    st.success("**Excellent! Your dataset is ready for model training.**")

# ===========================
# EXPORT
# ===========================
st.markdown("---")
st.subheader("Export Report")

report_df = dq.health.summary_report()
csv_report = report_df.write_csv()
st.download_button(
    "Download Summary Report (CSV)",
    data=csv_report,
    file_name="data_quality_report.csv",
    mime="text/csv",
)

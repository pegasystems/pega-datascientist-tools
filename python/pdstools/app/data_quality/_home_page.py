"""Home page for the Topic Data Quality app.

Exposes a single ``home_page()`` callable registered by
``_navigation.build_navigation()`` as the default page.
"""

from __future__ import annotations


def home_page() -> None:
    """Render the Topic Data Quality home page."""
    import polars as pl
    import streamlit as st

    from pdstools.utils.streamlit_utils import (
        set_active_app as _set_active_app,
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="Topic Data Quality")
    show_sidebar_branding("Topic Data Quality")
    _set_active_app("dq")

    show_version_header()

    st.markdown("# Topic Data Quality Checker")
    st.markdown(
        """
This tool helps you understand if your labelled text data is ready
for building a topic classification model. Upload a CSV, pick the
text and label columns, and run a comprehensive quality check.

**What it checks:**
- Class imbalance across topics
- Topic overlap (TF-IDF similarity)
- Text quality (duplicates, short texts)
- Cluster tightness and potential labelling errors (via embeddings)
"""
    )

    st.markdown("### Data import")

    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)",
        type=["csv"],
        key="dq_file_uploader",
    )

    if uploaded_file is not None:
        df = pl.read_csv(uploaded_file)
        st.success(f"Loaded **{df.height:,}** rows × **{len(df.columns)}** columns.")

        text_col = st.selectbox(
            "Select text column",
            df.columns,
            help="Column containing the text data you want to classify",
            key="dq_text_col",
        )
        topic_col = st.selectbox(
            "Select topic/label column",
            df.columns,
            help="Column containing the topic labels",
            key="dq_topic_col",
        )

        with st.sidebar:
            st.header("Analysis Options")
            st.markdown("---")
            similarity_threshold = st.slider(
                "Topic Overlap Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                help="Topics with similarity above this will be flagged as overlapping",
                key="dq_sim_threshold",
            )

        if st.button("Run Quality Check", type="primary"):
            from pdstools.data_quality import TopicDataQuality

            with st.spinner("Analysing your dataset – this may take a minute…"):
                dq = TopicDataQuality.from_dataframe(
                    df=df,
                    text_col=text_col,
                    topic_col=topic_col,
                    similarity_threshold=similarity_threshold,
                )
                # Pre-compute so sub-pages are instant
                dq.compute.embeddings()
                dq.compute.umap()
                dq.compute.topic_similarity()

                st.session_state["topic_dq"] = dq
                st.toast("Analysis complete – navigate to the Quality Report page.", icon="✅")
                st.rerun()

    # If analysis was already done, show a brief summary
    if "topic_dq" in st.session_state:
        dq = st.session_state["topic_dq"]
        health = dq.health.calculate_health_score()

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Health Score", f"{health['score']}/100")
        with col2:
            st.metric("Total Samples", f"{dq.total_samples:,}")
        with col3:
            st.metric("Unique Topics", dq.num_topics)
        with col4:
            st.metric("Balance Ratio", f"{dq.imbalance_ratio:.1f}x")

        st.info("Navigate to **Quality Report** in the sidebar for the full analysis.")
    elif uploaded_file is None:
        st.info("Upload a CSV file to get started.")

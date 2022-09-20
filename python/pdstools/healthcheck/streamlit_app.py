import streamlit as st
import logging
if __name__ == "__main__":
    from datamart_utils import import_data, generate_modeldata_filters
else:
    from .datamart_utils import import_data, generate_modeldata_filters


def run():
    import os

    st.write("# Generate a health check.")
    file_loc = os.path.dirname(__file__)
    params = dict()
    params["name"] = st.text_input("Customer name", "Sample")
    if " " in params["name"] or len(params["name"]) == 0:
        st.write("Please enter a valid name, without any spaces.")
        return None
    st.write("### Data settings")
    params, data = import_data(params, 1)
    if data is None:
        return None
    else:
        df = data.modelData
    params = generate_modeldata_filters(df, params)
    if "pandasquery" in params and len(params["pandasquery"]) > 0:
        filtereddf = df.query(params["pandasquery"])
    else:
        filtereddf = df
    st.write(f"Shape of model file: {filtereddf.shape}")
    if data.predictorData is not None:
        st.write(
        f'Shape of predictor file: {data.predictorData.query(f"ModelID in {list(filtereddf.ModelID.unique())}").shape}'
    )
    else:
        st.write('Could not find predictor data. Check if the file is uploaded correctly or if the file is in the correct folder.')


    st.markdown(
        """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )

    with st.expander("Export options", expanded=False):
        output_type = st.selectbox("Export format", ["pdf", "html", "docx"], index=1)
        st.write('Cache location')
        st.code(file_loc)
        output_location = file_loc
        filename = f'{output_location}/HealthCheck_{params["name"]}.{output_type}'
        param_file = f"{output_location}/params.yaml"
        persist_cache = st.checkbox("Keep cached files", value=False)

    st.write(params)

    if st.button("Generate healthcheck"):

        with st.spinner("Saving cached data..."):
            if "kwargs" not in params:
                params["kwargs"] = dict()
            import os

            params["kwargs"]["path"] = os.path.abspath(output_location)
            params["kwargs"]["model_filename"] = "models.parquet"
            if "pandasquery" in params:
                df = df.query(params.pop("pandasquery"))
            df.to_parquet(f"{output_location}/{params['kwargs']['model_filename']}")
            params["kwargs"]["include_cols"] = list(df.columns)
            if data.predictorData is not None:
                filtered_pred_data = data.predictorData.query(
                    f'ModelID in {list(df["ModelID"].unique())}'
                )
                params["kwargs"]["predictor_filename"] = "preds.parquet"
                filtered_pred_data.to_parquet(
                    f"{output_location}/{params['kwargs']['predictor_filename']}"
                )

        healthCheckDir = os.path.dirname(__file__)
        healthCheckName = "HealthCheck.qmd"
        bashCommand = f"quarto render {f'{healthCheckDir}/{healthCheckName}'} --to {output_type} --output {filename} --data-dir {os.path.abspath(f'{healthCheckDir}/HealthCheck_files/')} --execute-params {param_file}"
        params["Command"] = {
            "healthCheckDir": healthCheckDir,
            "healthCheckName": healthCheckName,
            "bashCommand": bashCommand,
        }
        add_params(param_file, params)

        with st.spinner("Generating healthcheck..."):
            import subprocess

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            
            logging.info(f"Process communicated: {process.communicate()}")
            if os.path.isfile(filename):
                st.session_state["file"] = open(filename, "rb")

                if not persist_cache:
                    with st.spinner("Removing cached files..."):
                        import os

                        to_remove = {
                            param_file,
                            f"{params['kwargs']['path']}/{params['kwargs']['model_filename']}",
                            filename,
                        }
                        if "predictor_filename" in params["kwargs"].keys():
                            to_remove = to_remove.union(
                                {
                                    f"{params['kwargs']['path']}/{params['kwargs']['predictor_filename']}"
                                }
                            )
                        for i in to_remove:
                            logging.info(f"Removing {i}")
                            os.remove(i)
    if 'file' in st.session_state:
        btn = st.download_button(
            label="Download file",
            data=st.session_state["file"],
            file_name=f'Healthcheck_{params["name"]}.{output_type}',
        )


def add_params(paramfile, params):
    logging.info("Saving yaml.")
    import yaml

    with open(paramfile, "w") as f:
        yaml.dump(params, f)


if __name__ == "__main__":
    run()

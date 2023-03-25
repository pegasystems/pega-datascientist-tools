import streamlit as st
import logging
import os
import shutil
import polars as pl

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
from pdstools.app.datamart_utils import import_data, generate_modeldata_filters

if __name__ == "__main__":
    from datamart_utils import import_data, generate_modeldata_filters
else:
    from .datamart_utils import import_data, generate_modeldata_filters


def run(**kwargs):
    import os

    st.title("Generate a Health Check :male-doctor:")
    st.write(
        """
             This application generates a Health Check that provides an overview of the Adaptive models and predictors. 
             
             You can simply run it by uploading your ADM data (see [pdstools wiki](https://github.com/pegasystems/pega-datascientist-tools/wiki/How-to-export-and-use-the-ADM-Datamart)
             to learn about ADM tables and how export them) and clicking on **Generate healthcheck** button.
             
             If you want to focus only on a subset of models/predictors, you can add filters and preview the filtered data.
             If you want to change model context keys, use edit parameters button.
             
             Before clicking on **Generate healthcheck** button, you can preview the data that will be used for creating the document.
            """
    )
    file_loc = os.path.dirname(__file__)
    params = dict()
    params["name"] = st.text_input("Customer name", "Sample")
    if " " in params or len(params) == 0:
        st.write("Please enter a valid name, without any spaces.")
        return None
    st.write("### Data settings")
    params, data = import_data(params, 1, **kwargs)
    if data is None:
        return None
    data.modelData = data.modelData.lazy()
    if data.predictorData is not None:
        data.predictorData = data.predictorData.lazy()

    filtereddf, params = generate_modeldata_filters(data, params)

    st.write(
        f"Shape of model file: {filtereddf.select(pl.count()).collect().item(), len(filtereddf.columns)}"
    )
    if data.predictorData is not None:
        filteredpreds = data.predictorData.join(
            filtereddf.select(pl.col("ModelID").unique()), on="ModelID", how="inner"
        )
        st.write(
            f"Shape of predictor file: {filteredpreds.select(pl.count()).collect().item(), len(filteredpreds.columns)}"
        )
    else:
        st.write(
            "Could not find predictor data. Check if the file is uploaded correctly or if the file is in the correct folder."
        )

    st.markdown(
        """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )
    with st.expander("Cache Location", expanded=False):
        cache_location = file_loc
        st.code(cache_location)
        params["name"] = params["name"].replace(" ", "_")
        output_filename = f'HealthCheck_{params["name"]}.html'
        cwd = os.getcwd()
        quarto_file_name = (
            "HealthCheck.qmd"
            if data.predictorData is not None
            else "HealthCheckModel.qmd"
        )
        param_file = f"{cache_location}/params.yaml"
        persist_cache = st.checkbox("Keep cached files", value=False)

    if st.button("Generate healthcheck"):
        with st.spinner("Saving cached data..."):
            if "kwargs" not in params:
                params["kwargs"] = dict()
            import os

            params["kwargs"]["path"] = os.path.abspath(cache_location)
            params["kwargs"]["model_filename"] = "models.arrow"
            filtereddf.collect().write_ipc(
                f"{cache_location}/{params['kwargs']['model_filename']}"
            )
            params["kwargs"]["include_cols"] = list(filtereddf.columns)
            if data.predictorData is not None:
                params["kwargs"]["predictor_filename"] = "preds.arrow"
                filteredpreds.collect().write_ipc(
                    f"{cache_location}/{params['kwargs']['predictor_filename']}"
                )

        try:
            shutil.copyfile(
                os.path.join(file_loc, quarto_file_name),
                os.path.join(cwd, quarto_file_name),
            )
            logging.info(
                f"copied quarto file {quarto_file_name} from {os.path.join(file_loc, quarto_file_name)} to {os.path.join(cwd, quarto_file_name)}"
            )
        except shutil.SameFileError:
            pass
        bashCommand = f"quarto render {f'{cwd}/{quarto_file_name}'} --to html --output {output_filename} --execute-params {param_file}"
        params["Command"] = {
            "quarto_file_name": quarto_file_name,
            "bashCommand": bashCommand,
        }
        add_params(param_file, params)

        with st.spinner("Generating healthcheck..."):
            import subprocess

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

            logging.info(f"Process communicated: {process.communicate()}")
            if os.path.isfile(output_filename):
                st.session_state["file"] = open(output_filename, "rb")

                if not persist_cache:
                    with st.spinner("Removing cached files..."):
                        import os

                        to_remove = {
                            param_file,
                            f"{params['kwargs']['path']}/{params['kwargs']['model_filename']}",
                            output_filename,
                        }
                        if "predictor_filename" in params["kwargs"].keys():
                            to_remove = to_remove.union(
                                {
                                    f"{params['kwargs']['path']}/{params['kwargs']['predictor_filename']}"
                                }
                            )
                        if file_loc != cwd:
                            to_remove.union({os.path.join(cwd, quarto_file_name)})

                        for i in to_remove:
                            logging.info(f"Removing {i}")
                            os.remove(i)
    if "file" in st.session_state:
        btn = st.download_button(
            label="Download file",
            data=st.session_state["file"],
            file_name=output_filename,
        )


def add_params(paramfile, params):
    logging.info("Saving yaml.")
    import yaml

    params["filters"] = None
    with open(paramfile, "w") as f:
        yaml.dump(params, f)


if __name__ == "__main__":
    run()

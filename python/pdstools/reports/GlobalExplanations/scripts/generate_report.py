import datetime
import json
import os

from pdstools.explanations import DataLoader, GradientBoostGlobalExplanations


def _process_data(data_location: str, model_name: str, end_date: datetime, output_folder: str):
    explanations = GradientBoostGlobalExplanations(
        data_folder=data_location,
        model_name=model_name,
        overwrite=True,
        end_date=end_date,
        output_folder=output_folder,
    )
    print(f"Processing data for model {model_name} until {end_date}")
    explanations.process()


def _get_context_string(context_info):
    return "-".join([v.replace(" ", "") for _, v in context_info.items()])


def _write_by_context_qmd(input_folder: str):
    data_loader = DataLoader(input_folder)
    contexts = data_loader.get_context_infos()
    all_contexts_file_name = "all_contexts.qmd"
    all_contexts_file_path = f"./by-context/{all_contexts_file_name}"
    with open(all_contexts_file_path, "w") as f:
        f.write(
            """
---
title: "By Context"
skip_exec: true
---

The top 10 predictor's contribtuions per context.

"""
        )

        f.write(
            f"""
```{{python}}
from pdstools.explanations import DataLoader, Plotter
import json
from IPython.display import display, Markdown
data_loader = DataLoader("{input_folder}")

```
"""
        )
        for context in contexts:
            context_string = _get_context_string(context)
            context_label = ('plt-' + context_string).lower()
            context_file_name = f"{context_label}.qmd"
            context_location = f"./by-context/{context_file_name}"
            with open(f"{context_location}", "w") as f_context:
                f.write(
                    f"""

The top 10 predictors contributions for the context `{context_string}`.
```{{python}}
#| label: {context_label}

plots = Plotter.plot_predictor_contributions(
        data_loader,
        {json.dumps(context)}
    )


for plot in plots:
    title = plot.layout['title']['text']
    display(Markdown(f'### {{title}}'), plot)

```
"""
                )

                # display(Markdown(f'### {{plot.layout["title"]['text']}}'), plot)

                f_context.write(
                    f"""
---
title: "{context_string}"
format: html
---

{{{{< embed {all_contexts_file_name}#{context_label} >}}}}
"""
                )


if __name__ == "__main__":#
    if not os.getenv("QUARTO_PROJECT_RENDER_ALL"):
        exit()

    DATA_FOLDER = os.getenv("DATA_FOLDER")
    MODEL_NAME = os.getenv("MODEL_NAME")
    TO_DATE = datetime.datetime.strptime(os.getenv("TO_DATE"), "%Y-%m-%d").date()

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    TMP_FOLDER = cwd + "/.tmp/out"
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    OUTPUT_FOLDER = cwd + "/by-context"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    _process_data(DATA_FOLDER, MODEL_NAME, TO_DATE, TMP_FOLDER)

    _write_by_context_qmd(TMP_FOLDER)

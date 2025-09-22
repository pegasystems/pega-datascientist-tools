import json
import os

import yaml

ENCODING = "utf-8"

CONTEXT_FOLDER = "by-context"

TOP_N = 10
TOP_K = 10
VERBOSE_DEFAULT = False
DATA_FOLDER = "aggregated_data"
UNIQUE_CONTEXTS_FILENAME = "unique_contexts.json"

PLOTS_FOR_BATCH = "plots_for_batch"
PARAMS_FILENAME = "params.yml"

# init template folder and filenames
# these are the templates used to generate the context files
# and the overview file
TEMPLATES_FOLDER = "./assets/templates"
INTRODUCTION_FILENAME = "getting-started.qmd"
OVERVIEW_FILENAME = "overview.qmd"
ALL_CONTEXT_HEADER_TEMPLATE = "all_context_header.qmd"
ALL_CONTEXT_CONTENT_TEMPLATE = "all_context_content.qmd"
SINGLE_CONTEXT_TEMPLATE = "context.qmd"


class ReportGenerator:
    def __init__(self):
        self.report_folder = os.getcwd()

        self.root_dir = ""
        self.data_folder = ""
        self.top_n = None
        self.top_k = None

        self.by_context_folder = f"{self.report_folder}/{CONTEXT_FOLDER}"
        if not os.path.exists(self.by_context_folder):
            os.makedirs(self.by_context_folder, exist_ok=True)

        self.plots_for_batch_filepath = f"{self.by_context_folder}/{PLOTS_FOR_BATCH}"
        self.contexts = None

        self._read_params()

    def _log_params(self):
        print(f"""
Report generation initialized with the following parameters:
- Aggregations folder: {self.data_folder}
- Report folder: {self.report_folder}
- Context folder: {self.by_context_folder}
- Plots for batch, filepath basename: {self.plots_for_batch_filepath}
- Top N: {self.top_n}
- Top K: {self.top_k}
        """)

    def _read_params(self):
        params_file = os.path.join(self.report_folder, "scripts", PARAMS_FILENAME)

        if not os.path.exists(params_file):
            self.top_n = TOP_N
            self.top_k = TOP_K
            self.verbose = VERBOSE_DEFAULT
            self.data_folder = DATA_FOLDER
            print(f"Parameters file {params_file} does not exist. Using defaults.")
        else:
            with open(params_file, "r", encoding=ENCODING) as file:
                params = yaml.safe_load(file)
                self.top_n = params.get("top_n", TOP_N)
                self.top_k = params.get("top_k", TOP_K)
                self.verbose = params.get("verbose", VERBOSE_DEFAULT)
                self.data_folder = params.get("data_folder", DATA_FOLDER)

        self.root_dir = os.path.abspath(os.path.join(self.report_folder, ".."))

        self.data_folder = os.path.abspath(
            os.path.join(self.report_folder, "..", self.data_folder)
        )
        print(f"Using data folder: {self.data_folder}")

        if self.verbose:
            self._log_params()

    @staticmethod
    def _get_context_dict(context_info: str) -> dict:
        return json.loads(context_info)["partition"]
    
    def _get_context_string(self, context_info: str) -> str:
        return "-".join([v.replace(" ", "") for _, v in self._get_context_dict(context_info).items()])

    @staticmethod
    def _read_template(template_filename: str) -> str:
        """Read a template file and return its content."""
        with open(f"{TEMPLATES_FOLDER}/{template_filename}", "r", encoding=ENCODING) as fr:
            return fr.read()

    def _write_single_context_file(
        self, 
        embed_path_for_batch: str, 
        filename: str, 
        template: str, 
        context_str: str, 
        context_label: str
    ):
        with open(filename, "w", encoding=ENCODING) as fw:
            f_context_template = f"""{
                template.format(
                    EMBED_PATH_FOR_BATCH=embed_path_for_batch,
                    CONTEXT_STR=context_str,
                    CONTEXT_LABEL=context_label,
                    TOP_N=self.top_n,
                )
            }"""
            fw.write(f_context_template)

    def _write_header_to_file(self, file_batch_nb: str, filename: str):
        template = self._read_template(ALL_CONTEXT_HEADER_TEMPLATE)

        f_template = f"""{
            template.format(
                ROOT_DIR=self.root_dir, 
                DATA_FOLDER=self.data_folder, 
                DATA_PATTERN=f"*_BATCH_{file_batch_nb}.parquet",
                TOP_N=self.top_n
            )
        }"""
        
        with open(filename, "w", encoding=ENCODING) as writer:
            writer.write(f_template)

    def _append_content_to_file(
        self,
        filename: str,
        template: str,
        context_dict: dict,
        context_label: str,
    ):
        with open(filename, "a", encoding=ENCODING) as writer:
            f_content_template = f"""{
                template.format(
                    CONTEXT_DICT=context_dict,
                    CONTEXT_LABEL=context_label,
                    TOP_N=self.top_n,
                    TOP_K=self.top_k,
                )
            }"""

            writer.write("\n")
            writer.write(f_content_template)
            
    def _get_batches(self):
        import glob
        batches = []
        for batch in glob.glob(f"{self.data_folder}/NUMERIC_BATCH_*.parquet"):
            batch_nb = os.path.basename(batch).split("_")[-1]
            batches.append(batch_nb.replace(".parquet", ""))
        return batches

    def _get_unique_contexts(self):
        if self.contexts is not None:
            return self.contexts
        
        unique_contexts_file = f"{self.data_folder}/{UNIQUE_CONTEXTS_FILENAME}"
        if not os.path.exists(unique_contexts_file):
            raise FileNotFoundError(
                f"Unique contexts file not found: {unique_contexts_file}. "
                "Please ensure that aggregates have been generated."
            )
        with open(unique_contexts_file, "r", encoding=ENCODING) as f:
            import json
            self.contexts = json.load(f)
        return self.contexts

    def _get_batch_filepath(self, file_batch_nb: str, with_sub_folders: bool = False) -> str:
        if with_sub_folders:
            plots_for_batch_folderpath = f"{self.plots_for_batch_filepath}_{file_batch_nb}"
            if not os.path.exists(plots_for_batch_folderpath):
                os.makedirs(plots_for_batch_folderpath, exist_ok=True)
            
            plots_for_batch_filename = f"{file_batch_nb}.qmd"
            plots_for_batch_filepath = f"{plots_for_batch_folderpath}/{plots_for_batch_filename}"
            return plots_for_batch_filepath
        else:
            return f"{self.plots_for_batch_filepath}_{file_batch_nb}.qmd"
            
    def _get_batch_embedpath(self, file_batch_nb: str, with_sub_folders: bool = False) -> str:
        if with_sub_folders:
            return f"{PLOTS_FOR_BATCH}_{file_batch_nb}/{file_batch_nb}.qmd"
        else:
            return f"{PLOTS_FOR_BATCH}_{file_batch_nb}.qmd"
        
    def _get_batch_filepath_for_single(self, context_label: str, file_batch_nb: str, with_sub_folders: bool = False) -> str:
        if with_sub_folders:
            plots_for_batch_folderpath = f"{self.plots_for_batch_filepath}_{file_batch_nb}"
            if not os.path.exists(plots_for_batch_folderpath):
                os.makedirs(plots_for_batch_folderpath, exist_ok=True)

            return f"{plots_for_batch_folderpath}/{context_label}.qmd"
        else:
            return f"{self.by_context_folder}/{context_label}.qmd"
            
    def _generate_by_context_qmds(self):
        
        with_sub_folders = False
        
        contexts = self._get_unique_contexts()
        
        for file_batch_nb, context_batches in contexts.items():
            print(f"Processing file batch: {file_batch_nb}")
            
            plots_for_batch_filepath = self._get_batch_filepath(file_batch_nb, with_sub_folders=with_sub_folders)
            
            # write header
            self._write_header_to_file(file_batch_nb, plots_for_batch_filepath)

            # write content
            context_content_template = self._read_template(ALL_CONTEXT_CONTENT_TEMPLATE)
            single_context_template = self._read_template(SINGLE_CONTEXT_TEMPLATE)

            for query_batch_nb, contexts in context_batches.items():
                for context in contexts:
                    context_str = self._get_context_string(context)
                    context_label = ("plt-" + context_str).lower()

                    self._append_content_to_file(
                        filename=plots_for_batch_filepath,
                        template=context_content_template,
                        context_dict=self._get_context_dict(context),
                        context_label=context_label,
                    )

                    self._write_single_context_file(
                        embed_path_for_batch=self._get_batch_embedpath(file_batch_nb, with_sub_folders=with_sub_folders),
                        filename=self._get_batch_filepath_for_single(context_label, file_batch_nb, with_sub_folders=with_sub_folders),
                        template=single_context_template,
                        context_str=context_str,
                        context_label=context_label,
                    )

    def _generate_overview_qmd(self):
        with open(f"{TEMPLATES_FOLDER}/{OVERVIEW_FILENAME}", "r", encoding=ENCODING) as fr:
            template = fr.read()

        f_template = f"""{
            template.format(
                ROOT_DIR=self.root_dir,
                DATA_FOLDER=self.data_folder,
                TOP_N=self.top_n,
                TOP_K=self.top_k,
            )
        }
        """

        with open(OVERVIEW_FILENAME, "w", encoding=ENCODING) as f:
            f.write(f_template)

    def _generate_introduction_qmd(self):
        with open(f"{TEMPLATES_FOLDER}/{INTRODUCTION_FILENAME}", "r", encoding=ENCODING) as fr:
            template = fr.read()

        f_template = f"""{
            template.format(
                TOP_N=self.top_n,
                TOP_K=self.top_k,
            )
        }"""

        with open(INTRODUCTION_FILENAME, "w", encoding=ENCODING) as f:
            f.write(f_template)

    def run(self):
        """Main method to generate the report files."""
        self._generate_introduction_qmd()
        print("Generated introduction QMD file.")

        self._generate_overview_qmd()
        print("Generated overview QMD file.")

        self._generate_by_context_qmds()
        print("Generated by-context QMDs files.")


if __name__ == "__main__":
    generator = ReportGenerator()
    generator.run()

import json
import os

import yaml

ENCODING = "utf-8"

CONTEXT_FOLDER = "by-context"

TOP_N = 10
TOP_K = 10
VERBOSE_DEFAULT = False
DATA_FOLDER = "aggregated_data"

ALL_CONTEXT_FILENAME = "all_contexts.qmd"
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

        self.all_context_filepath = f"{self.by_context_folder}/{ALL_CONTEXT_FILENAME}"

        self._read_params()

    def _log_params(self):
        print(f"""
Report generation initialized with the following parameters:
- Aggregations folder: {self.data_folder}
- Report folder: {self.report_folder}
- Context folder: {self.by_context_folder}
- All contexts file path: {self.all_context_filepath}
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
    def _get_context_string(context_info) -> str:
        return "-".join([v.replace(" ", "") for _, v in context_info.items()])

    @staticmethod
    def _read_template(template_filename: str) -> str:
        """Read a template file and return its content."""
        with open(f"{TEMPLATES_FOLDER}/{template_filename}", "r", encoding=ENCODING) as fr:
            return fr.read()

    def _write_single_context_file(
        self, filename: str, template: str, context_string: str, context_label: str
    ):
        with open(filename, "w", encoding=ENCODING) as fw:
            f_context_template = f"""{
                template.format(
                    ALL_CONTEXT_FILENAME=ALL_CONTEXT_FILENAME,
                    CONTEXT_STR=context_string,
                    CONTEXT_LABEL=context_label,
                    TOP_N=self.top_n,
                )
            }"""
            fw.write(f_context_template)

    def _write_header_to_file(self):
        template = self._read_template(ALL_CONTEXT_HEADER_TEMPLATE)

        f_template = f"""{
            template.format(
                ROOT_DIR=self.root_dir, DATA_FOLDER=self.data_folder, TOP_N=self.top_n
            )
        }"""

        with open(self.all_context_filepath, "w", encoding=ENCODING) as writer:
            writer.write(f_template)

    def _append_content_to_file(
        self,
        template: str,
        context_string: str,
        context_label: str,
        context: dict,
    ):
        with open(self.all_context_filepath, "a", encoding=ENCODING) as writer:
            f_content_template = f"""{
                template.format(
                    CONTEXT_STR=context_string,
                    CONTEXT_LABEL=context_label,
                    CONTEXT=json.dumps(context),
                    TOP_N=self.top_n,
                    TOP_K=self.top_k,
                )
            }"""

            writer.write("\n")
            writer.write(f_content_template)

    def _get_unique_contexts(self):
        unique_contexts_file = f"{self.data_folder}/unique_contexts.csv"
        if not os.path.exists(unique_contexts_file):
            raise FileNotFoundError(
                f"Unique contexts file not found: {unique_contexts_file}. "
                "Please ensure that aggregates have been generated."
            )
        with open(unique_contexts_file, "r", encoding=ENCODING) as f:
            lines = [line.strip() for line in f.readlines()]

        contexts = []
        for line in lines:
            if not line:
                continue
            context_info = json.loads(line)
            if not isinstance(context_info, dict):
                raise ValueError(
                    f"Invalid context format in {unique_contexts_file}: {line}"
                )
            contexts.append(context_info.get("partition", {}))

        return contexts

    def _generate_by_context_qmds(self):
        # write header
        self._write_header_to_file()

        # write content
        context_content_template = self._read_template(ALL_CONTEXT_CONTENT_TEMPLATE)
        single_context_template = self._read_template(SINGLE_CONTEXT_TEMPLATE)

        contexts = self._get_unique_contexts()
        for context in contexts:
            context_string = self._get_context_string(context)
            context_label = ("plt-" + context_string).lower()

            self._append_content_to_file(
                template=context_content_template,
                context_string=context_string,
                context_label=context_label,
                context=context,
            )

            single_context_filename = f"{self.by_context_folder}/{context_label}.qmd"

            self._write_single_context_file(
                filename=single_context_filename,
                template=single_context_template,
                context_string=context_string,
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

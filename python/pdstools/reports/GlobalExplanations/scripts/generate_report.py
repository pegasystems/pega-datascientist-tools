import json
import os
import yaml

from pdstools.explanations import ExplanationsDataLoader as DataLoader, ContextInfo

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
OVERVIEW_FILENAME = "overview.qmd"
ALL_CONTEXT_HEADER_TEMPLATE = "all_context_header.qmd"
ALL_CONTEXT_CONTENT_TEMPLATE = "all_context_content.qmd"
SINGLE_CONTEXT_TEMPLATE = "context.qmd"

class ReportGenerator:
    
    def __init__(self):
        
        self.report_folder = os.getcwd()
        
        self.data_folder = ""
        self.top_n = None
        self.top_k = None
        
        self.context_folder = f'{self.report_folder}/{CONTEXT_FOLDER}'
        if not os.path.exists(self.context_folder):
            os.makedirs(self.context_folder, exist_ok=True)
            
        self.all_context_filepath = f'{self.context_folder}/{ALL_CONTEXT_FILENAME}'
        
        self._read_params()
    
    def _log_params(self):
        print(f"""
Report generation initialized with the following parameters:
- Aggregations folder: {self.data_folder}
- Report folder: {self.report_folder}
- Context folder: {self.context_folder}
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
            with open(params_file, 'r') as file:
                params = yaml.safe_load(file)
                self.top_n = params.get('top_n', TOP_N)
                self.top_k = params.get('top_k', TOP_K)
                self.verbose = params.get('verbose', VERBOSE_DEFAULT)
                self.data_folder = params.get('data_folder', DATA_FOLDER)
        
        self.data_folder = os.path.abspath(os.path.join(self.report_folder, "..", self.data_folder))
                
        if self.verbose:
            self._log_params()
                
    @staticmethod
    def _get_context_string(context_info) -> str:
        return "-".join([v.replace(" ", "") for _, v in context_info.items()])

    @staticmethod
    def _read_template(template_filename: str) -> str:
        """Read a template file and return its content."""
        with open(f'{TEMPLATES_FOLDER}/{template_filename}', 'r') as fr:
            return fr.read()

    def _write_context_file(self,
                            filename: str, 
                            template: str, 
                            context_string: str, 
                            context_label: str):
        
        with open(filename, "w") as fw:
            f_context_template = f"""{
                template.format(
                    ALL_CONTEXT_FILENAME=ALL_CONTEXT_FILENAME,
                    CONTEXT_STR=context_string,
                    CONTEXT_LABEL=context_label,
                )
            }"""
            fw.write(f_context_template)
 
    @staticmethod           
    def _write_header_to_file(filename: str,
                            template: str,
                            input_folder: str):
        with open(filename, "w") as writer:
            writer.write(f"""{
                template.format(
                    DATA_FOLDER=input_folder)}""")
    
    @staticmethod        
    def _append_content_to_file(filename: str,
                            template: str,
                            context_string: str,
                            context_label: str,
                            context: ContextInfo):
        with open(filename, "a") as writer:
            f_content_template = f"""{
                template.format(
                    CONTEXT_STR=context_string,
                    CONTEXT_LABEL=context_label,
                    CONTEXT=json.dumps(context),)}
            """
            writer.write("\n")
            writer.write(f_content_template)

    def _generate_by_context_qmds(self):
        
        data_loader = DataLoader(self.data_folder)
        contexts = data_loader.get_context_infos()
        
        header_template = self._read_template(ALL_CONTEXT_HEADER_TEMPLATE)
        content_template = self._read_template(ALL_CONTEXT_CONTENT_TEMPLATE)
        context_template = self._read_template(SINGLE_CONTEXT_TEMPLATE)
        
        # write header
        self._write_header_to_file(
            filename=self.all_context_filepath,
            template=header_template,
            input_folder=self.data_folder
        )
        
        # write content
        for context in contexts:
            context_string = self._get_context_string(context)
            context_label = ('plt-' + context_string).lower()
            context_file_name = f"{context_label}.qmd"
            context_location = f"{self.context_folder}/{context_file_name}"
            
            self._append_content_to_file(
                filename=self.all_context_filepath,
                template=content_template,
                context_string=context_string,
                context_label=context_label,
                context=context
            )
            
            self._write_context_file(
                filename=context_location,
                template=context_template,
                context_string=context_string,
                context_label=context_label
            )

    def _generate_overview_qmd(self):
        
        with open(f'{TEMPLATES_FOLDER}/{OVERVIEW_FILENAME}', 'r') as fr:
            template = fr.read()
                
        f_template = f"""{template.format(
            DATA_FOLDER=self.data_folder,
            TOP_N=self.top_n,
            TOP_K=self.top_k,)}
        """
        
        with open(OVERVIEW_FILENAME, 'w') as f:
            f.write(f_template)

    def run(self):
        
        self._generate_overview_qmd()
        print("Generated overview QMD file.")
        
        self._generate_by_context_qmds()
        print("Generated by-context QMDs files.")
        
if __name__ == "__main__":
    # if not os.getenv("QUARTO_PROJECT_RENDER_ALL"):
    #     exit()
    generator = ReportGenerator()
    generator.run()


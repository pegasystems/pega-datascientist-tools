from ..base import UploadedModel as ModelUploaderPrevious


class UploadedModel(ModelUploaderPrevious):
    def __init__(self, repository_name: str, file_path: str):
        self.repository_name = repository_name
        self.file_path = file_path

    def __repr__(self):
        return "Model upload succesful. Use ChampionChallenger.add_model() to add this model to a prediction."

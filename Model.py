import fasttext


class Model:
    def __init__(self, model_name, model_path):
        """
        Class that stores the prediction model
        Parameters
        ----------
        model_name : string
            stores the name of the model
        model_path : string
            stores the path from where to load the model
        """
        self.model_name = model_name
        if self.model_name == 'FastText':
            self.model_path = model_path
            self.model = fasttext.load_model(model_path)
        else:
            self.model_path = ""
            self.model = None

    def predict(self, text):
        label_arr = self.model.predict(text)
        return label_arr[0][0], label_arr[1][0]

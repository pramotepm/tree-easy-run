import ConfigParser


class ParamLoad:
    def __init__(self):
        self.conf = None

        self.input_path = None
        self.true_class_path = None
        self.feature_name_path = None
        self.out_dir = None
        self.delim = None

        self.write_prob = None
        self.n_folds = None

        self.model_export_path = None
        self.unseen_sample_path = None

    def read_config(self, conf_file='conf/params.ini'):
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(conf_file)
        return self.__read_params()

    def __read_params(self):
        self.input_path = self.conf.get('Data', 'training_sample_file_path')
        self.true_class_path = self.conf.get('Data', 'class_file_path')
        self.feature_name_path = self.conf.get('Data', 'feature_name_file_path')
        self.out_dir = self.conf.get('Data', 'output_directory_path')
        self.delim = self.conf.get('Data', 'delimiter').decode("string_escape")

        self.n_folds = int(self.conf.get('Validation', 'n_folds'))

        self.model_export_path = self.conf.get('Prediction', 'model_export_path')
        self.unseen_sample_path = self.conf.get('Prediction', 'unseen_sample_file_path')
        self.unseen_sample_path = [self.unseen_sample_path, None][self.unseen_sample_path == 'None']
        return self

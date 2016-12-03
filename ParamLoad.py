import ConfigParser


class ParamLoad:
    def __init__(self):
        self.conf = None

        self.input_path = None
        self.class_path = None
        self.featu_path = None
        self.out_dir = None
        self.delim = None

        self.write_prob = None
        self.dup_sample_pred = None
        self.n_folds = None

    def read_config(self, conf_file='conf/params.ini'):
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(conf_file)
        return self.__read_params()

    def __read_params(self):
        self.input_path = self.conf.get('Data', 'training_sample_file_path')
        self.class_path = self.conf.get('Data', 'class_file_path')
        self.featu_path = self.conf.get('Data', 'feature_name_file_path')
        self.out_dir = self.conf.get('Data', 'output_directory_path')
        self.delim = self.conf.get('Data', 'delimiter').decode("string_escape")

        self.n_folds = int(self.conf.get('Validation', 'n_folds'))
        return self

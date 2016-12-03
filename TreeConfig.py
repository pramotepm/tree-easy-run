import ConfigParser
import numpy as np


class TreeConfig:
    def __init__(self):
        self.params = []
        self.model = None
        self.sample_path = None
        self.label_path = None
        self.output_dir = None

    def __parse2range(self, opts):
        param_range = []
        for op in opts.replace(' ', '').split(','):
            if ':' in op:
                start, end, step = op.split(':')
                if '.' in start or '.' in end or '.' in step:
                    start = float(start)
                    end = float(end)
                    step = float(step)
                    param_range.extend(np.arange(start, end + step, step))
                else:
                    start = int(start)
                    end = int(end)
                    step = int(step)
                    param_range.extend(range(start, end + 1, step))
            elif '.' in op:
                param_range.append(float(op))
            elif op.isdigit() or op.replace('-', '').isdigit():
                param_range.append(int(op))
            elif op == 'None':
                param_range.append(None)
            elif op == 'True':
                param_range.append(True)
            elif op == 'False':
                param_range.append(False)
            else:
                param_range.append(op)
        return param_range

    def __read_options(self, conf, section):
        options = conf.options(section)
        for option in options:
            yield option, conf.get(section, option)

    def __parse_sections(self, conf, sections):
        for section in sections:
            for option, val in self.__read_options(conf, section):
                val = self.__parse2range(val)
                yield option, val

    def conf_map(self, conf):
        model_param = {}
        algo = conf.get('Model', 'model')

        if algo == 'RandomForestClassifier' or algo == 'ExtraTreesClassifier':
            sections = ['TreeBase', 'ForestSpecific', 'Miscellaneous']
        elif algo == 'GradientBoosting':
            sections = ['TreeBase', 'GradientSpecific', 'Miscellaneous']
        else:
            pass  # error

        for k, v in self.__parse_sections(conf, sections):
            model_param[k] = v

        # No parameter 'n_jobs' in GradientBoosting
        if algo == 'GradientBoosting':
            model_param.pop('n_jobs')

        self.model = algo
        self.params = model_param

    def read_config(self, conf_file='./conf/tree_params.ini'):
        config = ConfigParser.ConfigParser()
        config.read(conf_file)
        self.conf_map(config)
        return self

if __name__ == '__main__':
    tc = TreeConfig().read_config()
    print tc.model
    print tc.params
    print tc.sample_path
    print tc.label_path
    print tc.output_dir
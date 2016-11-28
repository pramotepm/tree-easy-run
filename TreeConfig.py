import ConfigParser


def __read_options(conf, section):
    options = conf.options(section)
    for option in options:
        yield option, conf.get(section, option)


def __read_tree_base(conf):
    for option, val in __read_options(conf, 'TreeBase'):
        if option == 'max_features':
            if val == 'sqrt' or val == 'auto' or val == 'log2':
                pass
            elif val == 'None':
                val = None
            else:
                try:
                    _ = float(val)
                except ValueError:
                    pass
                val = float(val) if '.' in val else int(val)
        elif option == 'max_depth':
            val = None if val == 'None' else int(val)
        else:
            val = float(val) if '.' in val else int(val)
        yield option, val


def __read_forest_option(conf):
    for option, val in __read_options(conf, 'ForestSpecific'):
        if option == 'criterion':
            if not (val == 'gini' or val == 'entropy'):
                pass
        if option == 'oob_score' or option == 'bootstrap':
            if not (val == 'True' or val == 'true' or val == 'False' or val == 'false'):
                pass  # error
            else:
                val = True if val == 'True' or val == 'true' else False
        yield option, val


def __read_gradient_option(conf):
    for option, val in __read_options(conf, 'GradientSpecific'):
        try:
            val = float(val)
        except ValueError:
            pass
        yield option, val


def __read_misc(conf, algo):
    for option, val in __read_options(conf, 'Miscellaneous'):
        if algo == 'GradientBoosting' and option == 'n_jobs':
            continue
        else:
            val = int(val)
        yield option, val


def conf_map(conf):
    model_param = {}
    algo = conf.get('Model', 'model')
    for k, v in __read_tree_base(conf):
        model_param[k] = v

    if algo == 'RandomForestClassifier' or algo == 'ExtraTreesClassifier':
        for k, v in __read_forest_option(conf):
            model_param[k] = v

    elif algo == 'GradientBoosting':
        for k, v in __read_gradient_option(conf):
            model_param[k] = v

    for k, v in __read_misc(conf, algo):
        model_param[k] = v

    else:
        pass  # error

    return algo, model_param


def read_config(filename='./conf/tree_params.ini'):
    config = ConfigParser.ConfigParser()
    config.read(filename)
    return conf_map(config)

if __name__ == '__main__':
    print read_config()

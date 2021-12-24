import argparse
import json
import os
import itertools
import copy
import tempfile
import yaml
import pprint
import argparse

# Global configuration values for default output and storage.
STORAGE_ROOT = ".."
ENV_SETUP_SCRIPT = os.path.join("setup_shell.sh")
TMP_DIR = os.path.join(STORAGE_ROOT, "tmp")
FOLDER_KEYS = []
DEFAULT_ENTRY_POINT = "scripts/train.py"
DEFAULT_REQUIRED_ARGS = ["path", "config"]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", type=str, default=DEFAULT_ENTRY_POINT)
    parser.add_argument("--arguments", metavar="KEY=VALUE", nargs='+', help="Set kv pairs used as args for the entry point script.")
    parser.add_argument("--jobs-per-instance", type=int, default=1)
    return parser

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def get_jobs(args):
    script_args = parse_vars(args.arguments)
    # Handle the default case, train.
    if args.entry_point == DEFAULT_ENTRY_POINT:
        '''
        Custom code for sweeping using the experiment sweeper.
        '''
        for arg_name in DEFAULT_REQUIRED_ARGS:
            assert arg_name in script_args

        if script_args['config'].endswith(".json"):
            experiment = Experiment.load(script_args['config'])
            configs_and_paths = [(c, os.path.join(script_args['path'], n)) for c, n in experiment.generate_configs_and_names()]
        else:
            configs_and_paths = [(script_args['config'], script_args['path'])]

        jobs = [{"config": c, "path" : p} for c, p in configs_and_paths]
        for arg_name in script_args.keys():
            if not arg_name in jobs[0]:
                print("Warning: argument", arg_name, "being added globally to all python calls with value", script_args[arg_name])
                for job in jobs:
                    job[arg_name] = script_args[arg_name]

    else:
        # we have the default configuration
        jobs = [script_args.copy() for _ in range(args.jobs_per_instance)]
        if args.jobs_per_instance:
            # We need some way of distinguishing the jobs, so set the seed argument
            # Scripts must implement this if they want to be able to run multiple on the same machine
            for i in range(args.jobs_per_instance):
                seed = jobs[i].get('seed', 0)
                jobs[i]['seed'] = int(seed) + i

    return jobs

class Config(object):
    '''
    A lightweight copy of the config file with only basic IO capabilities.
    This is used so that we don't load in the full package on slurm head nodes.
    This is a bit of a work around for now, but it saves a lot of time.
    '''

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def update(self, d):
        self.config.update(d)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def copy(self):
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config

class Experiment(dict):

    def __init__(self, base=None, name=None, paired_keys=[]):
        super().__init__()
        self._name = name
        self.base_config = Config.load(base)
        self.paired_keys = paired_keys

    @property
    def name(self):
        return self._name

    @classmethod
    def load(cls, path):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as fp:
            data = json.load(fp)
        # Run formatting checks
        assert 'base' in data, "Did not supply a base config"
        base_config = data['base']
        del data['base'] # Remove the base configuration

        if 'paired_keys' in data:
            # We have some paired values. This means that in the variant updater these are all changed at the same time.
            paired_keys = data['paired_keys']
            assert isinstance(paired_keys, list)
            if len(paired_keys) > 0:
                assert all([isinstance(key_pair, list) for key_pair in paired_keys])
            del data['paired_keys']
        else:
            paired_keys = []

        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, list)
        experiment = cls(base=base_config, name=name, paired_keys=paired_keys)
        experiment.update(data)
        return experiment

    def get_variants(self):
        paired_keys = set()
        for key_pair in self.paired_keys:
            for k in key_pair:
                if k in paired_keys:
                    raise ValueError("Key was paired multiple times!")
                paired_keys.add(k)
        
        groups = []
        unpaired_keys = [key for key in self.keys() if not key in paired_keys] # Fix the ordering!
        unpaired_variants = itertools.product(*[self[k] for k in unpaired_keys])
        unpaired_variants = [{key:variant[i] for i, key in enumerate(unpaired_keys)} for variant in unpaired_variants]
        groups.append(unpaired_variants)

        # Now construct the paired variants
        for key_pair in self.paired_keys:
            # instead of using product, use zip
            pair_variant = zip(*[self[k] for k in key_pair]) # This gets all the values
            pair_variant = [{key:variant[i] for i, key in enumerate(key_pair)} for variant in pair_variant]
            groups.append(pair_variant)

        group_variants = itertools.product(*groups)
        # Collapse the variants, making sure to copy the dictionaries so we don't get duplicates
        variants = []
        for variant in group_variants:
            collapsed_variant = {k:v for x in variant for k,v in x.items()}
            variants.append(collapsed_variant)

        return variants

    def generate_configs_and_names(self):
        variants = self.get_variants()
        configs_and_names = []
        for i, variant in enumerate(variants):
            config = self.base_config.copy()
            name = ""
            remove_trailing_underscore = False
            for k, v in variant.items():
                config_path = k.split('.')
                config_dict = config
                while len(config_path) > 1:
                    if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                    config_dict = config_dict[config_path[0]]
                    config_path.pop(0)
                if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                config_dict[config_path[0]] = v
                
                if k in FOLDER_KEYS:
                    name = os.path.join(v, name)
                elif len(self[k]) > 1:
                    # Add it to the path name if it is different for each run.
                    if isinstance(v, str):
                        str_val = v
                    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or v is None:
                        str_val = str(v)
                    elif isinstance(v, list):
                        str_val = '_'.join([str(val) for val in v])                        
                    else:
                        raise ValueError("Could not convert config value to str.")

                    name += str(config_path[0]) + '-' + str_val + '_'
                    remove_trailing_underscore = True

            if remove_trailing_underscore:
                name = name[:-1]
            name = os.path.join(self.name, name)
            if not os.path.exists(TMP_DIR):
                os.mkdir(TMP_DIR)
            _, config_path = tempfile.mkstemp(text=True, prefix='config', suffix='.json', dir=TMP_DIR)
            print("Variant", i+1)
            print(config)
            config.save(config_path)
            configs_and_names.append((config_path, name))
        
        return configs_and_names
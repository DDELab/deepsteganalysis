from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import full

def deep_dict_merge(yaml_dict, cli_dict):
    if (type(yaml_dict) is DictConfig):
        new_dict = dict(yaml_dict)
    else:
        new_dict = cli_dict
        return new_dict

    for key in cli_dict.keys():
        if type(cli_dict[key]) is DictConfig and key in yaml_dict.keys():
            new_dict[key] = deep_dict_merge(yaml_dict[key], cli_dict[key])
        elif type(cli_dict[key]) is not DictConfig and key in yaml_dict.keys():
            new_dict[key] = cli_dict[key]
        elif key in yaml_dict.keys():
            new_dict[key] = yaml_dict[key]
        else:
            new_dict[key] = cli_dict[key]
    return new_dict

def clean_empty_keys(conf):
    to_be_deleted = []
    for key in conf.keys():
        if conf[key] and type(conf[key]) is DictConfig:
            conf[key] = clean_empty_keys(conf[key])
        if conf[key] == None:
            to_be_deleted.append(key)

    for key in to_be_deleted:
        del conf[key]
        
    return conf

def clean_cli_conf(cli_conf):
    for key in cli_conf.keys():
        if "--" in key:
            new_key = key[2:]
            cli_conf[new_key] = cli_conf[key]
            del cli_conf[key]
    return cli_conf

def get_args_cli_yaml(path=""):
    cfg_path = ""
    conf_cli = OmegaConf.from_cli()
    conf_cli = clean_cli_conf(conf_cli)

    for key in conf_cli.keys():
        if key == "config":
            cfg_path = conf_cli[key]
            break

    if cfg_path == "" and path == "":
        raise ValueError("No configarion file specified!")
    elif cfg_path != "":
        del conf_cli[key]
    else:
        cfg_path = path

    conf_yaml = OmegaConf.load(cfg_path)
    full_conf = deep_dict_merge(conf_yaml, conf_cli)
    full_conf = clean_empty_keys(full_conf)
    full_conf = OmegaConf.create(full_conf)

    #print(OmegaConf.to_yaml(full_conf))
    return full_conf

if __name__ == '__main__':
    args = get_args_cli_yaml(path="cfg/debug.yaml")
    print(OmegaConf.to_yaml(args))
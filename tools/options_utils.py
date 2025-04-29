from omegaconf import OmegaConf

def get_args_cli_yaml(cfg_path=""):
    conf_cli = OmegaConf.from_cli()

    if "config" in conf_cli.keys():
        cfg_path = conf_cli["config"]

    if cfg_path == "":
        raise ValueError("No configarion file specified!")

    conf_yaml = OmegaConf.load(cfg_path)
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_yaml, conf_cli)
    conf = OmegaConf.create(conf)
    
    return conf

def conf2dict(conf):

    def conf2dict_(conf):
        new_dict = {}
        if type(conf) != dict:
            return conf
        for k, v in conf.items():
            new_dict[k] = conf2dict_(v)
        return new_dict

    conf = OmegaConf.to_container(conf, resolve=True)
    return conf2dict_(conf)

if __name__ == '__main__':
    args = get_args_cli_yaml(cfg_path="cfg/default.yaml")
    print(OmegaConf.to_yaml(args))
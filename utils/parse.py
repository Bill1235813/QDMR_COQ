import configargparse as cfargparse
import json
import sys
import os
import opts as opts


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    def parse_with_config(self):
        opt = self.parse_args()
        if opt.config is not None:
            config_args = json.load(open(opt.config))
            override_keys = {
                arg[2:].split("=")[0] for arg in sys.argv[1:] if
                arg.startswith("--")
            }
            for k, v in config_args.items():
                if k not in override_keys:
                    setattr(opt, k, v)
        return opt

    def save_config(self, opt):
        save_model_path = os.path.abspath(opt.save_model)
        model_dirname = os.path.dirname(save_model_path)
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)
        if opt.save_config is not None:
            with open("%s/config.txt" % opt.save_model, "w") as f:
                f.write(self.format_values())

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = opts.model_opts(cls()).parse_known_args([])[0]
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

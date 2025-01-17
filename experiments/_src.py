import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
DB_DIR = config["paths"]["dbs_dir"]
sys.path.append(PROJECT_DIR)

from src.objectives.warcraft import WarcraftObjective, ConstraintWarcraft, get_map
from src.objectives.ackley import ackley
from src.objectives.sphere import sphere
from src.objectives.eggholder import Eggholder
from src.samplers.custom_botorch import CustomBoTorchSampler
from src.samplers.parafac import ParafacSampler
from src.samplers.parafac_nonneg import NonnegParafacSampler
from src.samplers.tucker import TuckerSampler
from src.samplers.tf import TFSampler
from src.samplers.tf_continual import TFContinualSampler
from src.samplers.tf_continual_v2 import TFContinualSamplerV2
from src.utils_experiments import set_logger, parse_experiment_path


__all__ = [
    "WarcraftObjective",
    "ConstraintWarcraft",
    "get_map",
    "ackley",
    "sphere",
    "Eggholder",
    "CustomBoTorchSampler",
    "ParafacSampler",
    "NonnegParafacSampler",
    "TuckerSampler",
    "TFSampler",
    "TFContinualSampler",
    "TFContinualSamplerV2",
    "set_logger",
    "parse_experiment_path",
    "ParafacSamplerV2",
]

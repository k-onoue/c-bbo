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

from src.objectives.warcraft import ConstraintWarcraft, get_map
from src.objectives.warcraft import WarcraftObjectiveTF
from src.objectives.warcraft import WarcraftObjectiveBenchmark
from src.objectives.eggholder import EggholderBenchmark, EggholderTF
from src.objectives.ackley import AckleyBenchmark, AckleyTF
from src.samplers.tf_continual import TFContinualSampler
from src.samplers.gp import GPSampler
from src.utils_experiments import set_logger, parse_experiment_path


__all__ = [
    "WarcraftObjectiveTF",
    "WarcraftObjectiveBenchmark",
    "ConstraintWarcraft",
    "EggholderBenchmark",
    "EggholderTF",
    "AckleyBenchmark",
    "AckleyTF",
    "TFContinualSampler",
    "GPSampler",
    "set_logger",
    "get_map",
    "parse_experiment_path",
]
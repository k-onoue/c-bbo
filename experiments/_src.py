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
from src.samplers.tf_sdpa import TFSdpaSampler
from src.samplers.gp import GPSampler
from src.utils_experiments import set_logger, parse_experiment_path

from src.objectives.diabetes import DiabetesObjective
from src.objectives.pressure import PressureVesselObjective
from src.samplers.tf_continual_ablation import TFContinualAblationSampler

__all__ = [
    "WarcraftObjectiveTF",
    "WarcraftObjectiveBenchmark",
    "ConstraintWarcraft",
    "EggholderBenchmark",
    "EggholderTF",
    "AckleyBenchmark",
    "AckleyTF",
    "TFContinualSampler",
    "TFSdpaSampler",
    "GPSampler",
    "set_logger",
    "get_map",
    "parse_experiment_path",
    "DiabetesObjective",
    "PressureVesselObjective"
    "TFContinualAblationSampler",
]
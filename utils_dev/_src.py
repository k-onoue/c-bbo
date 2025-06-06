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
from src.objectives.ackley import AckleyTF
from src.objectives.diabetes import DiabetesObjective


__all__ = [
    "WarcraftObjectiveTF",
    "ConstraintWarcraft",
    "AckleyTF",
    "get_map",
    "DiabetesObjective"
]
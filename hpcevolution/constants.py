from pathlib import Path
ACTIVE_AGENTS_DIR_PATHLIB = Path(__file__).parent / 'active_agents'
WORK_DIR_PATHLIB = Path(__file__).parent / 'work_todo'
RESULTS_DIR_PATHLIB = Path(__file__).parent / 'results'

ACTIVE_AGENTS_DIR = str(ACTIVE_AGENTS_DIR_PATHLIB)

ACTIVE_EXTENSION = '.active'
MODEL_EXTENSION = '.model'

SLEEP_TIME = 5
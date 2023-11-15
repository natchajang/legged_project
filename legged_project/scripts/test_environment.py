# append root path of project to find all module in project
import sys
sys.path.append("/home/natcha/github/legged_project")

from legged_project.envs import *
from legged_project.utils import get_args, task_registry

if __name__ == '__main__':
    print(task_registry.env_cfgs)
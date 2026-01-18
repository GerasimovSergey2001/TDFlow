from collections import OrderedDict
import numpy as np
from custom_dmc_tasks import point_mass_maze
import shimmy

def force_set_state(vec_env, target_state):
    """
    target_state: np.array([x, y, vx, vy])
    """
    new_obs_list = []
    
    for e in vec_env.envs:
        dmc_env = e.unwrapped._env 
        dmc_env.task.set_state(dmc_env.physics, target_state)
        obs_dict = dmc_env.task.get_observation(dmc_env.physics)
        flat_obs = np.concatenate([obs_dict['position'], obs_dict['velocity']])
        new_obs_list.append(flat_obs)
    new_obs_list = np.array(new_obs_list)
    return OrderedDict({'position':new_obs_list[:, :2], 'velocity': new_obs_list[:, 2:]})

def get_true_physics_state(vec_env, env_idx=0):
    physics = vec_env.envs[env_idx].unwrapped.physics
    
    true_pos = physics.data.qpos[:2].copy()
    true_vel = physics.data.qvel[:2].copy()
    
    return OrderedDict({'position':true_pos, 'velocity': true_vel})

def make_env(task):
    raw_env = point_mass_maze.make(task=task)
    return shimmy.DmControlCompatibilityV0(raw_env)
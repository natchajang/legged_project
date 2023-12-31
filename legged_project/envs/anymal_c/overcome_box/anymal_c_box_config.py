# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
from legged_project.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AnymalCBoxCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 52 # default is 235 measure height = 187 242
        num_envs = 4096       # number of environment default = 4096
        num_actions = 12      # number of action equal to Dof (control with actuator network)
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        env_spacing = 3.      # not used with heightfields/trimeshes
        
        num_steps_per_env = 24 # use for save log in each iteration (need to equal to nnum_steps_per_env in train config)
        
    class terrain( LeggedRobotCfg.terrain ):
        # measure terrain
        measure_heights = False # get the measurement for being in obs
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain type
        mesh_type = 'plane' # ['plane', 'box']
        terrain_kwargs = {'num_obs':{'max':30, 'step':50}, 'obs_height':{'max':0.25, 'step':0.05}, 
                          'obs_width':{'max':0.5, 'step':0.1}, 'obs_length':{'max':2, 'step':0.4}} 
                        # Dict of arguments for selected terrain
                        # num_ods is number of obstacle
                        # dimension is in meters
                        # max and step need to relate to number of terrain row (level)

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.05 # [m]
        border_size = 5. # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        # for rough terrain except "plane" type
        curriculum = False # for increase difficulty of obstcle (box size)
        selected = False # select a unique terrain type and pass all arguments
   
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 20. # length of terrain per 1 level
        terrain_width = 20. # width of terrain per 1 level
        num_rows= 6 # number of terrain rows (levels) 
        num_cols = 1 # number of terrain cols (types)
        
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, -1] #Add type 8 (-1) for custom discreate terrain
        # trimesh only:
        slope_treshold = 0 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,

            "LF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RF_HFE": 0.4,
            "RH_HFE": -0.4,

            "LF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RF_KFE": -0.8,
            "RH_KFE": 0.8,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}       # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = False
        push_robots = False
        randomize_base_mass = False # make it's easier for learning
        added_mass_range = [-5., 5.]
        
    class commands( LeggedRobotCfg.commands ):
        curriculum = True # If true the tracking reward is above 80% of the maximum, increase the range of commands
        max_curriculum = 1.
        
        # limit update command range (use if curriculum is True)
        step_vel = 0.1
        step_height = 0.025
        step_angle = 0.02*math.pi
        max_vel = 1.3
        min_height = 0.25
        max_angle = 0.2*math.pi
        
        num_commands = 6 # default: 1.lin_vel_x, 2.lin_vel_y,
                         #          3.base_height
                         #          4.base_roll 5.base_pitch 6.base_yaw
        resampling_time = 10.   # time before command are changed [sec] 
                                # if do not want to resample during episode set more than env.episode_length_s
        heading_command = False # if true: compute ang vel command from heading error (not use in our task)
        
        class ranges:           # range of command
            # lin_vel_x = [-1.3, 1.3] # min max [m/s]
            # lin_vel_y = [-1.3, 1.3]   # min max [m/s]
            # base_height = [0.25, 0.50] # min max [m]
            # base_roll = [-0.2*math.pi, 0.2*math.pi]   # min max [rad]
            # base_pitch = [-0.2*math.pi, 0.2*math.pi]  # min max [rad]
            # base_yaw = [-0.2*math.pi, 0.2*math.pi]    # min max [rad]
            
            start_vel = 0.5
            start_height = 0.4
            start_angle = 0.06
            lin_vel_x = [-start_vel, start_vel] # min max [m/s]
            lin_vel_y = [-start_vel, start_vel]   # min max [m/s]
            base_height = [start_height, 0.50] # min max [m]
            base_roll = [-start_angle*math.pi, start_angle*math.pi]   # min max [rad]
            base_pitch = [-start_angle*math.pi, start_angle*math.pi]  # min max [rad]
            base_yaw = [-start_angle*math.pi, start_angle*math.pi]    # min max [rad]
        
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        # Set reward tracking function types
        reward_tracking = 'gaussian' # str ['binary', 'progress_estimator', 'gaussian']
        reward_tracking_accept = {'velocity': 0.3, 'height':0.02, 'orientation':0.03*math.pi}
        # Need to tune
        tracking_sigma = 0.1 # tracking reward = exp(-error^2/sigma) for linear velocity
        tracking_height = 0.01 # tracking reward = exp(-error^2/sigma) for height
        tracking_orientation = 0.01 # tracking reward = exp(-error^2/sigma) for base orientation
        
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.4 # not use for our code which tracking from command
        max_contact_force = 500. # forces above this value are penalized
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # pre defined reward function
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_acc = -2.5e-7
            action_rate = -0.01
            feet_air_time = 1.0
            collision = -1.
            
            # add my own reward functions
            tracking_lin_vel = 1.0
            tracking_height = 1.0
            tracking_orientation = 1.0
            
            # unenble some reward functions
            action_rate = 0
            termination = -0.0
            tracking_ang_vel = 0.
            base_height = 0. # unused fix base height reward
            stand_still = -0.
            feet_stumble = -0.0
            dof_vel = -0.
            orientation = -0.
            
class AnymalCBoxCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    
    class algorithm ( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1000 # number of policy updates
        
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = 'ex2_with_curriculum'               # sub experiment of each domain => save as name of folder
        experiment_name = 'anymal_c_box' # domain of experiment
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
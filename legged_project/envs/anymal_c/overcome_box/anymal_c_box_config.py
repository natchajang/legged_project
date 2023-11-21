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

from legged_project.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AnymalCBoxCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 512     # number of environment default = 4096
        num_actions = 12    # number of action equal to Dof (control with actuator network)

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = True
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, -1] #Add type 8 (-1) for custom discreate terrain
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

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
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
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

    class domain_rand( LeggedRobotCfg.domain_rand):
        # randomize_base_mass = True
        randomize_base_mass = False # make it's easier for learning
        added_mass_range = [-5., 5.]
        
    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 6 # default: 1.lin_vel_x, 2.lin_vel_y,
                         #          3.base_height
                         #          4.base_roll 5.base_pitch 6.base_yaw
        resampling_time = 10. # time before command are changed [sec]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            base_height = [] # min max [m]
            base_roll = []   # min max [rad]
            base_pitch = []  # min max [rad]
            base_yaw = []    # min max [rad]
        
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.4
        max_contact_force = 500. # forces above this value are penalized
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # pre defined reward function
            tracking_lin_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_acc = -2.5e-7
            feet_air_time =  1.0
            collision = -1. 
            action_rate = -0.01
            
            # add my own reward functions
            tracking_height = 1.0
            tracking_oreintation = 1.0
            
            # unenble some reward functions
            tracking_ang_vel = 0.
            base_height = 0. # unused fix base height reward
            stand_still = -0.
            feet_stumble = -0.0
            dof_vel = -0.
            orientation = -0.

class AnymalCBoxCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'box_anymal_c'
        load_run = -1
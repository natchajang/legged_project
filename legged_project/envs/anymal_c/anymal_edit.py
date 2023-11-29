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

from time import time
import numpy as np
import math
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_project.envs import LeggedRobot
from legged_project import LEGGED_GYM_ROOT_DIR
from .overcome_box.anymal_c_box_config import AnymalCBoxCfg
from legged_project.utils.math import quat_apply_yaw

class AnymalEdit(LeggedRobot):
    cfg : AnymalCBoxCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self.description_name = cfg.description.name
        self.commands_viz = False

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_mean_height[:] = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.base_quat[:] = self.root_states[:, 3:7]
        base_euler = torch.stack(list(get_euler_xyz(self.base_quat)), dim=1)
        self.base_euler[:] = torch.where(base_euler>math.pi, (-2*math.pi)+base_euler, base_euler)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            if self.debug_viz:
                self._draw_debug_vis()
            if self.commands_viz:
                self._draw_commands_vis()
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize buffer for obs and compute reward
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.height_measurements, 
                                            self.obs_scales.dof_pos,
                                            self.obs_scales.dof_pos,
                                            self.obs_scales.dof_pos], device=self.device, requires_grad=False,) 
        
        base_euler = torch.stack(list(get_euler_xyz(self.base_quat)), dim=1)
        self.base_euler = torch.where(base_euler>math.pi, (-2*math.pi)+base_euler, base_euler)
        self.base_mean_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) 
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  (self.base_mean_height * self.obs_scales.height_measurements).unsqueeze(1),
                                    self.base_euler * self.obs_scales.dof_pos,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # random command linear velocity (m/s) and base height (m)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # random roll pitch yaw command in range -x to x rad
        roll_command = torch_rand_float(self.command_ranges["base_roll"][0], self.command_ranges["base_roll"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        pitch_command = torch_rand_float(self.command_ranges["base_pitch"][0], self.command_ranges["base_pitch"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        yaw_command = torch_rand_float(self.command_ranges["base_yaw"][0], self.command_ranges["base_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # add orientation to command buffer
        self.commands[env_ids, 3] = roll_command
        self.commands[env_ids, 4] = pitch_command
        self.commands[env_ids, 5] = yaw_command
        
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # change range of linear velocity of x axis
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
    
    def _draw_commands_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        plane_geom_command = gymutil.WireframeBoxGeometry(2, 2, 0.005, None, color=(0, 1, 0))
        plane_geom_measure = gymutil.WireframeBoxGeometry(2, 2, 0.005, None, color=(1, 1, 0))
        axes_geom = gymutil.AxesGeometry(scale=1, pose=None)
        for i in range(self.num_envs):
            # measurement
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            base_quat = (self.root_states[i, 3:7]).cpu().numpy()
            base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])[0].cpu().numpy() # ref to base frame
            x = base_pos[0]
            y = base_pos[1]
            z = base_pos[2]
            rx = base_quat[0]
            ry = base_quat[1]
            rz = base_quat[2]
            rw = base_quat[3]
            vx = base_lin_vel[0]
            vy = base_lin_vel[1]
            vz = base_lin_vel[2]
            
            # commands
            command_height = self.commands[i, 2]
            command_vel = self.commands[i, :2].cpu().numpy()
            command_orien = self.commands[i, 3:]
            command_quat = quat_from_euler_xyz(command_orien[0], command_orien[1], command_orien[2])
      
            # Draw plane height
            plane_pose_measure = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(plane_geom_measure, self.gym, self.viewer, self.envs[i], plane_pose_measure)
            plane_pose_command = gymapi.Transform(gymapi.Vec3(x, y, command_height), r=None)
            gymutil.draw_lines(plane_geom_command, self.gym, self.viewer, self.envs[i], plane_pose_command)
            
            # Draw linear velocity direction
            # measurement
            vel_transform = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(rx, ry, rz, rw))
            vel_measure = vel_transform.transform_point(gymapi.Vec3(vx, vy, vz))
            line = [x, y, z+0.3, vel_measure.x, vel_measure.y, vel_measure.z+0.3]
            line_color = [1, 0, 1]
            self.gym.add_lines(self.viewer, self.envs[i], 1, line, line_color)
            # command
            vel_transform_command = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(rx, ry, rz, rw))
            vel_target = vel_transform_command.transform_point(gymapi.Vec3(command_vel[0], command_vel[1], 0))
            line = [x, y, z+0.3, vel_target.x, vel_target.y, vel_target.z+0.3]
            line_color = [0, 0.5, 0.5]
            self.gym.add_lines(self.viewer, self.envs[i], 1, line, line_color)
            
            # Draw orientation
            axes_pose = gymapi.Transform(gymapi.Vec3(x, y, z), gymapi.Quat(command_quat[0], command_quat[1], command_quat[2], command_quat[3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], axes_pose)
    
    def _command_tracking(self):
            roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
    
    
    #------------my additional reward functions----------------
    def _reward_tracking_height(self):
        # Penalize base height away from target which random for each iteration
        # command order => lin_vel_x, lin_vel_y, base_height, roll, pitch, yaw
        height_error = torch.square(self.commands[:, 2] - self.base_mean_height)
        return torch.exp(-height_error/self.cfg.rewards.tracking_height)
    
    def _reward_tracking_orientation(self):
        # Penalize base orientation away from target while walking
        # command order => lin_vel_x, lin_vel_y, base_height, roll, pitch, yaw
        orientation_error = torch.sum(torch.square(self.commands[:, 3:] - self.base_euler), dim=1) # error between command and measurement
        return 1 - torch.exp(-orientation_error/(self.cfg.rewards.tracking_orientation))
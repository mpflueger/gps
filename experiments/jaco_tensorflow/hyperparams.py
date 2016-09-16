""" Hyperparameters for PR2 policy optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.jaco.agent_jaco import AgentJaco
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import example_tf_network


EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, -0.05],
                      [0.02, 0.05, 0.0]])

#EE_POINT_TARGET = np.array([0, 0.4, 0.1, 0.1, 0.4, 0.1, 0, 0.5, 0.1])
#EE_POINT_TARGET = np.array([0.286, -0.043, 0.414, 0.349, 0.028, 0.383, 0.292,
#  -0.089, 0.325])
#EE_POINT_TARGET = np.array([-0.091, -0.500, 0.098, -0.191, -0.495, 0.102, -0.096, -0.599, 0.101])
EE_POINT_TARGET = np.array([-0.020, -0.534, 0.102, -0.112, -0.574, 0.111, 0.020, -0.626, 0.106])

SENSOR_DIMS = {
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    # JOINT_TORQUES: 6,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_JACOBIANS: 3 * EE_POINTS.shape[0] * 6,
    ACTION: 6,
}

PR2_GAINS = np.array([3.09, 1.08, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/jaco_tensorflow/'

common = {
    'experiment_name': 'jaco_tensorflow' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

x0s = []
ee_tgts = []
reset_conditions = []

# Set up each condition.
for i in xrange(common['conditions']):

    # Initial conditions for trial arm
    # ja_x0, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
    #     common['target_filename'], 'trial_arm', str(i), 'initial'
    # )
    # Initial conditions for aux arm
    # ja_aux, _, _ = load_pose_from_npz(
    #     common['target_filename'], 'auxiliary_arm', str(i), 'initial'
    # )
    # Target conditions for trial arm (as end effector pose)
    # _, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
    #     common['target_filename'], 'trial_arm', str(i), 'target'
    # )

    # Jaco home position at 0 velocity
    x0 = np.zeros(12 + 9)
    # x0[:6] = ja_x0[:6]
    x0[:6] = [4.80, 2.91, 0.99, 4.19, 1.43, 1.31] # home position
    # x0[:6] = [4.0453, 5.176, 0.0456, -2.876, 8.169, -1.386] #gripper down
    # x0[12:(12+9)] = np.ndarray.flatten(
    #     get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T
    # )

    # ee_tgt = np.ndarray.flatten(
    #     get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    # )

    aux_x0 = np.zeros(6)
    # aux_x0[:] = ja_aux[:6]

    reset_condition = {
        TRIAL_ARM: {
            'mode': JOINT_SPACE,
            'data': x0[0:6],
        },
        AUXILIARY_ARM: {
            'mode': JOINT_SPACE,
            'data': aux_x0,
        },
    }

    x0s.append(x0)
    # ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentJaco,
    'dt': 0.01,
    'conditions': common['conditions'],
    'T': 500,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    # 'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
    #                   END_EFFECTOR_POINT_VELOCITIES],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'end_effector_points': EE_POINTS,
    # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
    #                 END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'meta_include': [END_EFFECTOR_POINT_JACOBIANS],
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 10,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'init_pol_wt': 0.01,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    'final_weight': 50,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-3 / PR2_GAINS,
}

fk_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    # 'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'target_end_effector': EE_POINT_TARGET,
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

fk_cost2 = {
    'type': CostFK,
    # 'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'target_end_effector': EE_POINT_TARGET,
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1,1,1,1,1,1]),
            'target_state': np.array([4.61, 3.63, 1.97, 5.74, 1.72, -0.49])
            # 'target_state': np.array([4.5, 4.05, 1.45, 6.27, 0.61, 0.40]),
        },
        JOINT_VELOCITIES: {
            'wp': np.array([2,2,2,2,2,2]),
            'target_state': np.array([0,0,0,0,0,0]),
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost1, fk_cost2],
    # 'costs': [torque_cost, state_cost],
    'weights': [1.0, 1.0, 1.0],
    # 'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'iterations': 1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = generate_experiment_info(config)

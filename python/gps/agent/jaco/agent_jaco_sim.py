""" This file defines an agent for a Jaco arm, controlled via ROS """
import numpy as np
from threading import Lock
import time

import rospy

import actionlib
import controller_manager_msgs.srv
from jaco_msgs.msg import ArmJointAnglesAction, ArmJointAnglesGoal
from jaco_torque_driver_msgs.srv import SetControlMode
import std_msgs.msg
import sensor_msgs.msg
import tf
import trajectory_msgs.msg

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_JACOBIANS, ACTION
from gps.sample.sample import Sample

robot_command_topic = '/jaco/jaco2_controller/command'
robot_state_topic = '/jaco/joint_states'
joint_names = ['jaco2_joint_1', 'jaco2_joint_2', 'jaco2_joint_3',
               'jaco2_joint_4', 'jaco2_joint_5', 'jaco2_joint_6']

class AgentJacoSim(Agent):
    """
    Agent for a simulated Jaco robot
    TODO
    """
    def __init__(self, hyperparams, init_node=True):
        Agent.__init__(self, hyperparams)
        self._pos = {}
        self._vel = {}

        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
        self.x0 = self._hyperparams["x0"]

        # Create a ROS node so we can publish and subscribe
        if init_node:
            rospy.init_node('gps_agent_jaco')

        # Publisher for sending commands to robot
        self._command_pub = rospy.Publisher(
                robot_command_topic, std_msgs.msg.Float64MultiArray,
                queue_size=1)

        # Subscribe to robot state
        self._joint_state_sub = rospy.Subscriber(
                robot_state_topic, sensor_msgs.msg.JointState, self.joint_state_callback)
        self._joint_state_lock = Lock()
        self._joint_state_msg = sensor_msgs.msg.JointState()
        self._joint_state_fresh = False

        # Subscribe to end effector point jacobians
        self._ee_jacobian_sub = rospy.Subscriber(
                "/jaco/ee_jacobian", std_msgs.msg.Float64MultiArray, self.ee_jacobian_callback)
        self._ee_jacobian_lock = Lock()
        self._ee_jacobian_msg = std_msgs.msg.Float64MultiArray()

        # Setup out Transform Listener
        self._tf_listener = tf.TransformListener()


    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample
        TODO
        """
        print("sample called")

        self.switch_traj_control()

        # Move robot to initial position (specified by condition?)
        initial_condition_traj = trajectory_msgs.msg.JointTrajectory()
        initial_condition_traj.joint_names = [
            "jaco2_joint_1", "jaco2_joint_2", "jaco2_joint_3",
            "jaco2_joint_4", "jaco2_joint_5", "jaco2_joint_6"]
        x0_point = trajectory_msgs.msg.JointTrajectoryPoint()
        x0_point.time_from_start = rospy.Duration(secs=2)
        x0_point.positions = self.x0[condition][:6]
        initial_condition_traj.points.append(x0_point)
        trajectory_pub = rospy.Publisher("/jaco/jaco2_traj_controller/command",
                                         trajectory_msgs.msg.JointTrajectory,
                                         queue_size=1)
        trajectory_pub.publish(initial_condition_traj)
        time.sleep(2.5)

        # Get initial position off the robot
        self._joint_state_lock.acquire()
        x = self.joint_state_msg_to_state(self._joint_state_msg)
        #x = {JOINT_ANGLES: [0, 0, 0, 0, 0, 0], JOINT_VELOCITIES: [0, 0, 0, 0, 0, 0]}
        self._joint_state_lock.release()

        # Get the positions of our end effector points
        (ee_pt1, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_1", rospy.Time())
        (ee_pt2, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_2", rospy.Time())
        (ee_pt3, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_3", rospy.Time())
        x[END_EFFECTOR_POINTS] = ee_pt1 + ee_pt2 + ee_pt3

        # Get the jacobians for the end effector points
        self._ee_jacobian_lock.acquire()
        x[END_EFFECTOR_POINT_JACOBIANS] = np.reshape(
            self._ee_jacobian_msg.data,
            [self._ee_jacobian_msg.layout.dim[0].size, self._ee_jacobian_msg.layout.dim[1].size])
        self._ee_jacobian_lock.release()

        #sample = self._init_sample(x)

        # Run policy and save as a sample
        sample = self.run_policy(policy, x, noisy)

        if save:
            print("Saving policy Sample!")
            self._samples[condition].append(sample)
        return sample

    def run_policy(self, policy, x, noisy=True):
        """
        Run a policy on the robot, and return the sample.

        Returns:
                object: Sample object from the policy rollout
        """
        print("run_policy called")
        sample = self._init_sample(x)

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        # noise = np.zeros(6)

        self.switch_effort_control()

        # start_time = time.time()
        t = 0
        last_tick = time.time()
        # while time.time() - start_time <= time_to_run:
        while t < (self._hyperparams['T']):
            self._joint_state_lock.acquire()
            if (self._joint_state_fresh
                    and time.time() - last_tick >= self._hyperparams['dt']):
                last_tick = time.time()

                # Grab the state from the protected variable
                # obs = self.joint_state_msg_to_obs(self._joint_state_msg)
                x = self.joint_state_msg_to_state(self._joint_state_msg)
                self._joint_state_lock.release()

                # Get the positions of our end effector points
                (ee_pt1, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_1", rospy.Time())
                (ee_pt2, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_2", rospy.Time())
                (ee_pt3, _) = self._tf_listener.lookupTransform("/jaco2_link_base", "/jaco_ee_point_3", rospy.Time())
                x[END_EFFECTOR_POINTS] = ee_pt1 + ee_pt2 + ee_pt3

                obs = x

                # Get the jacobians for the end effector points
                m = {}
                self._ee_jacobian_lock.acquire()
                m[END_EFFECTOR_POINT_JACOBIANS] = np.reshape(
                        self._ee_jacobian_msg.data,
                        [self._ee_jacobian_msg.layout.dim[0].size, self._ee_jacobian_msg.layout.dim[1].size])
                self._ee_jacobian_lock.release()

                # Calculate and send new command
                u = policy.act(self.vectorize_x(x), self.vectorize_x(obs), t, noise[t, :])
                self._send_command(u)

                # Save this sample
                self.set_sample(sample, x, t)
                self.set_sample(sample, m, t)
                sample.set(ACTION, np.array(u), t)
                t += 1
            else:
                self._joint_state_lock.release()
                rospy.sleep(0.001)

        # Return to position control
        self.switch_traj_control()

        return sample

    def switch_traj_control(self):
        # Switch to the trajectory controller for position control
        sc_req = controller_manager_msgs.srv.SwitchControllerRequest()
        sc_req.start_controllers = ["jaco2_traj_controller"]
        sc_req.stop_controllers = ["jaco2_controller"]
        sc_client = rospy.ServiceProxy(
            "/jaco/controller_manager/switch_controller",
            controller_manager_msgs.srv.SwitchController)
        sc_client(sc_req)

    def switch_effort_control(self):
        # Switch to the effort controller for position control
        sc_req = controller_manager_msgs.srv.SwitchControllerRequest()
        sc_req.start_controllers = ["jaco2_controller"]
        sc_req.stop_controllers = ["jaco2_traj_controller"]
        sc_client = rospy.ServiceProxy(
            "/jaco/controller_manager/switch_controller",
            controller_manager_msgs.srv.SwitchController)
        sc_client(sc_req)

    def joint_state_callback(self, message):
        """
        Callback for new robot observation available.
        """
        self._joint_state_lock.acquire()
        self._joint_state_msg = message
        self._joint_state_fresh = True
        self._joint_state_lock.release()

    def ee_jacobian_callback(self, message):
        self._ee_jacobian_lock.acquire()
        self._ee_jacobian_msg = message
        self._ee_jacobian_lock.release()

    def _send_command(self, U):
        """
        Send a control command to the robot
        """
        # Build effort control message
        joint_efforts_msg = std_msgs.msg.Float64MultiArray()
        joint_efforts_msg.layout.data_offset = 0
        joint_efforts_msg.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        joint_efforts_msg.layout.dim[0].label = "joints"
        joint_efforts_msg.layout.dim[0].size = len(U)
        joint_efforts_msg.layout.dim[0].stride = 1
        for i in range(len(U)):
            joint_efforts_msg.data.append(U[i])

        # Send Message
        self._command_pub.publish(joint_efforts_msg)

    @staticmethod
    def joint_state_msg_to_obs(msg):
        obs = np.array([])
        for i in range(6):
            obs = np.append(obs, msg.position[i])
        for i in range(6):
            obs = np.append(obs, msg.velocity[i])
        return obs

    def joint_state_msg_to_state(self, msg):
        # Update position and velocity for the joints we care about
        # NB: this uses old values if they are not updated
        for i in range(len(msg.name)):
            if msg.name[i] in joint_names:
                self._pos[msg.name[i]] = msg.position[i]
                self._vel[msg.name[i]] = msg.velocity[i]

        # Put states in the correct order in our state vectors
        x = {}
        x[JOINT_ANGLES] = []
        x[JOINT_VELOCITIES] = []
        for joint in joint_names:
            x[JOINT_ANGLES].append(self._pos[joint])
            x[JOINT_VELOCITIES].append(self._vel[joint])

        return x

    @staticmethod
    def vectorize_x(x):
        x_vec = np.array([])
        for sensor in x.keys():
            x_vec = np.append(x_vec, x[sensor])
        return x_vec

    def _init_sample(self, x):
        sample = Sample(self)
        for sensor in x.keys():
            sample.set(sensor, np.array(x[sensor]), t=0)
        return sample

    @staticmethod
    def set_sample(sample, x, t):
        for sensor in x.keys():
            sample.set(sensor, np.array(x[sensor]), t=t)

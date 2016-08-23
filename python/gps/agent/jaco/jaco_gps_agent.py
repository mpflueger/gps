""" This file defines an agent for a Jaco arm, controlled via ROS """
import numpy as np
from threading import Lock
import time

import rospy

from gps.sample.sample import Sample
import std_msgs.msg
import sensor_msgs.msg

from gps.agent.agent import Agent
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES

robot_command_topic = '/jaco/jaco2_controller/command'
robot_state_topic = '/jaco/joint_states'

class AgentJaco(Agent):
    """
    Agent for a (maybe simulated) Jaco robot
    TODO
    """
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)

        # Publisher for sending commands to robot
        self._command_pub = rospy.Publisher(
                robot_command_topic, std_msgs.msg.Float64MultiArray,
                queue_size=1)

        # Subscribe to robot state
        self._joint_state_sub = rospy.Subscriber(
                robot_state_topic, sensor_msgs.msg.JointState, self._joint_state_callback)
        self._joint_state_lock = Lock()
        self._joint_state_fresh = False


    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample
        TODO
        """

        # TODO Move robot to initial position (specified by condition)

        x = {JOINT_ANGLES: [0, 0, 0, 0, 0, 0], JOINT_VELOCITIES: [0, 0, 0, 0, 0, 0]}
        #sample = self._init_sample(x)

        # Run policy and save as a sample
        sample = self.run_policy(policy, x)

        if save:
            self._samples[condition].append(sample)
        return sample

    def run_policy(self, policy, x, time_to_run=5):
        """
        Run a policy on the robot, and return the sample.

        Returns:
                object: Sample object from the policy rollout
        """
        sample = self._init_sample(x)

        start_time = time.time()
        t = 0
        while time.time() - start_time <= time_to_run:
            self._joint_state_lock.acquire()
            if self._joint_state_fresh:
                obs = self.joint_state_msg_to_obs(self._joint_state_msg)
                self._joint_state_lock.release()
                u = policy.act(None, obs, None, None)
                self._send_command(u)
                self.set_sample(sample, obs, t)
                t += 1
            else:
                self._joint_state_lock.release()
                rospy.sleep(0.001)

        return sample

    def _joint_state_callback(self, message):
        """
        Callback for new robot observation available.
        """
        self._joint_state_lock.acquire()
        self._joint_state_msg = message
        self._joint_state_fresh = True
        self._joint_state_lock.release()

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
        obs = []
        for i in range(6):
            obs.append(msg.position[i])
        for i in range(6):
            obs.append(msg.velocity[i])
        return obs

    def _init_sample(self, x):
        sample = Sample(self)
        for sensor in x.keys():
            sample.set(sensor, np.array(x[sensor]), t=0)

    @staticmethod
    def set_sample(sample, x, t):
        for sensor in x.keys():
            sample.set(sensor, np.array(x[sensor]), t=t+1)

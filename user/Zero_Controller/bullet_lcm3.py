import pybullet as p
import pybullet_data
# import lcm
# from lcmt.leg_control_command_lcmt import leg_control_command_lcmt
# from lcmt.leg_control_data_lcmt import leg_control_data_lcmt
# from lcmt.state_estimator_lcmt import state_estimator_lcmt
import numpy as np
import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], 'Cheetah-Gym'))
from dog import Dog
# , quat_to_YZX
import time
import zmq

# DYN_CONFIG = {'lateralFriction_toe': 1, # 0.6447185797960826, 
# 			  'lateralFriction_shank': 0.737, #0.6447185797960826  *  (0.351/0.512),
# 			  'contactStiffness': 4173, #2157.4863390669952, 
# 			  'contactDamping': 122, #32.46233575737161, 
# 			  'linearDamping': 0.03111169082496665, 
# 			  'angularDamping': 0.04396695866661371, 
# 			  'jointDamping': 0.03118494025640309, 
# 			  # 'w_y_offset': 0.0021590823485152914
# 			  }

DYN_CONFIG = {'lateralFriction_toe': 0.7, #0.8, # 0.6447185797960826, 
			  'lateralFriction_shank': 0.5, #0.737, #0.6447185797960826  *  (0.351/0.512),
			  'contactStiffness': 4173, #2729, #1530, #2157.4863390669952, 
			  'contactDamping': 200, #414, #160, #32.46233575737161, 
			  'linearDamping': 0.03111169082496665, 
			  'angularDamping': 0.04396695866661371, 
			  'jointDamping': 0.03118494025640309, 
			  'jointDamping': 0.03118494025640309, 
			  "max_force": 130,
			  "mass_body": 10.5,
			  "maxJointVelocity": 100
			  # 'w_y_offset': 0.0021590823485152914
			  }

ENV_ARGS = {"render": True, "real_time": True, "immortal": True, 
			"version": 3, "normalised_abduct": True, "debug_tuner_enable" : True,
			"mode": "sleep", "state_mode": "body_arm_leg_full", "custom_dynamics": DYN_CONFIG,
			"fix_body": True}

front_hip_offset = 0.11
back_hip_offset = -0.155;

lcm_state = np.zeros(12)
bullet_state = np.zeros(12)


class Sim(object):
	def __init__ (self):

		

		self.env = Dog(**ENV_ARGS)
		self.env.set_dynamics(DYN_CONFIG)
		# self.env.mode = "stand"
		p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

		self.env.startpoint = [0, 0, 0.00]
		self.env.startOrientation = p.getQuaternionFromEuler([0,0,0])


		# if hasattr(self, "stateId"):
		# 	self._p.restoreState(self.stateId)
		# else:
		# 	self.stateId = self._p.saveState()

		self.env._p.resetBasePositionAndOrientation(self.env.dogId, [0, 0, 0.0], p.getQuaternionFromEuler([0,0,0]))
		# self._p.resetBaseVelocity(self.dogId, [0]*3, [0]*3)


		# self.lc = lcm.LCM()
		# subscription = self.lc.subscribe("leg_control_command", self.receive_command)
		# subscription = self.lc.subscribe("debug_channel", self.update_debug_data)
		# subscription = self.lc.subscribe("leg_control_data", self.update_data)
		# subscription = self.lc.subscribe("state_estimator", self.update_data_body)
		self.walk_iter = -1;

		context = zmq.Context()

		#  Socket to talk to server
		print("Connecting to hello world server…")
		self.socket = context.socket(zmq.REQ)
		self.socket.connect("tcp://localhost:5555")

		self.phase = -1



	def _get_state(self):
			
		joints_state = p.getJointStates(dogId, motor_ids)
		joints_pos = [joints_state[i][0] for i in range(12)]
		torso_state = p.getBasePositionAndOrientation(dogId)
		torso_pos = [torso_state[0][i] for i in range(3)]
		height = torso_pos[2]
		torso_ori = [torso_state[1][i] for i in range(4)]

		get_velocity = p.getBaseVelocity(dogId)
		get_invert = p.invertTransform(torso_state[0], torso_state[1])
		get_matrix = p.getMatrixFromQuaternion(get_invert[1])
		torso_vel = [get_matrix[0] * get_velocity[1][0] + get_matrix[1] * get_velocity[1][1] + get_matrix[2] * get_velocity[1][2],
					 get_matrix[3] * get_velocity[1][0] + get_matrix[4] * get_velocity[1][1] + get_matrix[5] * get_velocity[1][2],
					 get_matrix[6] * get_velocity[1][0] + get_matrix[7] * get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]]

		# joint_pos_lite = [joints_pos[1], joints_pos[2], joints_pos[7],  joints_pos[8]]
		# joint_pos_lite = [joints_pos[1], joints_pos[7],  joints_pos[8], joints_pos[10], joints_pos[11]]
		joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4]]
		state = np.array([height] + torso_ori + torso_vel + joint_pos_lite)

		x = torso_pos[0]

		return state, x


	def update_debug_data(self, channel, data):
		msg = state_estimator_lcmt.decode(data)
		self.walk_iter = msg.p[0]
		if self.walk_iter == 0:
			print("START WALKING !!!!!!!!")
			q = [float("{:.3f}".format(i)) for i in self.env.lcm_state.quat]
			print("QUAT: ", q)

			
			# for i in range(10):
			# 	self.publish_data()
			# 	time.sleep(1)
			


			rpy = quat_to_YZX([self.env.lcm_state.quat[1], self.env.lcm_state.quat[2], self.env.lcm_state.quat[3], self.env.lcm_state.quat[0]])
			print("")
			print("PITCH AFTER RESET:::::::::::::::::::: ", rpy[1])
			# quit()


	def receive_command(self, channel, data):
		msg = leg_control_command_lcmt.decode(data)
		# print("Received message on channel \"%s\"" % channel)
		# print("   q_des   = %s" % str(msg.q_des))
		if not np.all(np.array(msg.kp_joint) == 0):
			actions = [i for i in msg.q_des]
			actions[1] -= front_hip_offset
			actions[4] -= front_hip_offset
			actions[7] -= back_hip_offset
			actions[10] -= back_hip_offset
			print("[DEBUG] ACTION TO JOIN[3][1]: ", actions[10])
			self.env.step_jpos(actions)
			print("[DEBUG] POSITION TO JOIN[3][1]: ", self.env.get_joins_pos()[10])
			
		else:
			self.env.step_jpos()

		# self.publish_data()

	def publish_data(self):

		self.env.update_lcm_state()

		msg = state_estimator_lcmt()
		msg.p = self.env.lcm_state.p
		msg.vWorld = self.env.lcm_state.vWorld
		msg.vBody = self.env.lcm_state.vBody
		msg.vRemoter = self.env.lcm_state.vRemoter
		msg.rpy = self.env.lcm_state.rpy
		msg.omegaBody = self.env.lcm_state.omegaBody
		msg.omegaWorld = self.env.lcm_state.omegaWorld
		msg.quat = self.env.lcm_state.quat
		msg.aBody = self.env.lcm_state.aBody
		msg.aWorld = self.env.lcm_state.aWorld
		self.env.get_full_state()
		print("PITCH: ", self.env.pitch)
		rpy = quat_to_YZX([msg.quat[1], msg.quat[2], msg.quat[3], msg.quat[0]])
		print("[DEBUG] QUAT2RPY: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  -->  [{:.4f}, {:.4f}, {:.4f}]".format(msg.quat[0], msg.quat[1], msg.quat[2], msg.quat[3], rpy[0], rpy[1], rpy[2]))
		self.lc.publish("state_estimator_bullet", msg.encode())

		msg = leg_control_data_lcmt()
		leg_pos = self.env.lcm_state.q
		leg_pos[1] += front_hip_offset
		leg_pos[4] += front_hip_offset
		leg_pos[7] += back_hip_offset
		leg_pos[10] += back_hip_offset
		# print("[DEBUG] POSITION OF JOIN[0][1] and JOIN[1][1] FROM BULLET: ", leg_pos[1], ", ", leg_pos[4])


		msg.q = leg_pos
		msg.qd = self.env.lcm_state.qd
		msg.p = self.env.lcm_state.p_leg
		
		msg.v = self.env.lcm_state.v
		msg.tau_est = self.env.lcm_state.tau_est
		self.lc.publish("leg_control_data_bullet", msg.encode())



	def update_data(self, channel, data):
		msg = leg_control_data_lcmt.decode(data)
		lcm_state[8] = msg.q[0]
		lcm_state[9] = msg.q[1]
		lcm_state[10] = msg.q[3]
		lcm_state[11] = msg.q[4]


	def update_data_body(self, channel, data):
		msg = state_estimator_lcmt.decode(data)
		lcm_state[0] = msg.p[2]

		lcm_state[1] = msg.quat[1]
		lcm_state[2] = msg.quat[2]
		lcm_state[3] = msg.quat[3]
		lcm_state[4] = msg.quat[0]

		lcm_state[5] = msg.omegaBody[0]
		lcm_state[6] = msg.omegaBody[1]
		lcm_state[7] = msg.omegaBody[2]

		formatted_lcm_state = [ float('%.2f' % elem) for elem in lcm_state ]

		# print("LCM STATE:    ", formatted_lcm_state)

		bullet_state = _get_state()[0]
		formatted_bullet_state = [ float('%.2f' % elem) for elem in bullet_state ]
		# print("BULLET STATE: ", formatted_bullet_state)

	def run_bak(self):
		try:
			while True:
				self.lc.handle()
				self.publish_data()
		except KeyboardInterrupt:
			pass

	def s_robot_offset(self, s):
		s_ = s
		s_[7] += front_hip_offset
		s_[9] += front_hip_offset
		s_[10] += back_hip_offset
		s_[12] += back_hip_offset

		return s_

	def a_robot_offset(self, a):
		a_ = a
		a_[1] -= front_hip_offset
		a_[4] -= front_hip_offset
		a_[7] -= back_hip_offset
		a_[10] -= back_hip_offset

		return a_


	def run(self):

		
		s = self.env.reset()
		s = self.s_robot_offset(s)
		t = 0

		actions = []

		while True:
			s_str = ""
			for s_i in s:
				s_str += "{:.4f}".format(s_i)
				s_str += ", "
			s_str = s_str[:-2]
			print("Sending request ", s_str)
			self.socket.send_string(s_str)
			#  Get the reply.
			message = self.socket.recv()
			action_str = message.decode("utf-8")
			print("Received Action ",  action_str)
			phase = int(action_str[6]) if action_str[6] != '-' else -1
			if self.phase != phase:
				self.phase = phase
				t = 0

			try:
				action_str_val = action_str[8:]
				jpos_action = [float(i) for i in action_str_val.split(", ")]
			except ValueError:
				action_str_val = action_str[9:]
				jpos_action = [float(i) for i in action_str_val.split(", ")]

			if np.all(np.array([float("{:.3f}".format(i)) for i in jpos_action])==0):
				jpos_action = None
			if jpos_action is not None:
				jpos_action = self.a_robot_offset(jpos_action)
			
			# if (self.phase == 4 and t > 1999):
			# 	self.env.mode = "stand"
			# 	self.env.reset()
			# else:
			# 	self.env.step_jpos(jpos_action)
			if self.phase >= 0 and self.phase < 5:
				actions.append(jpos_action)


			self.env.step_jpos(jpos_action)
			
			if self.phase == 5:
				time.sleep(0.002)
				# quit()
			if self.phase == 4:
				print("JPOS[0][1]: ", jpos_action)
				print("RESULTED [0][1]: ", self.env.get_joins_pos()[1])

			s, _, _ = self.env._get_state()
			s = self.s_robot_offset(s)
			# print("----QUAT: ", self.env.quat)
			print("----SENT JPOS[0][1]: ", s[7])

			t += 1
			
		
		# np.save("reset_actions", actions)



def quat_to_YZX(quat):

	def compose( position, quaternion, scale ):
		te = []
		x = quaternion[0]
		y = quaternion[1]
		z = quaternion[2]
		w = quaternion[3]
		x2 = x + x
		y2 = y + y
		z2 = z + z
		xx = x * x2
		xy = x * y2
		xz = x * z2
		yy = y * y2
		yz = y * z2
		zz = z * z2
		wx = w * x2
		wy = w * y2
		wz = w * z2
		sx = scale[0]
		sy = scale[1]
		sz = scale[2]
		te.append(( 1 - ( yy + zz ) ) * sx)
		te.append(( xy + wz ) * sx)
		te.append(( xz - wy ) * sx)
		te.append(0)
		te.append(( xy - wz ) * sy)
		te.append(( 1 - ( xx + zz ) ) * sy)
		te.append(( yz + wx ) * sy)
		te.append(0)
		te.append(( xz + wy ) * sz)
		te.append(( yz - wx ) * sz)
		te.append( ( 1 - ( xx + yy ) ) * sz)
		te.append(0)
		te.append( position[0])
		te.append( position[1])
		te.append( position[2])
		te.append(1)

		return te

	mat = compose([0]*3, quat, [1]*3)
	m11 = mat[0]
	m12 = mat[4]
	m13 = mat[8]
	m21 = mat[1]
	m22 = mat[5]
	m23 = mat[9]
	m31 = mat[2]
	m32 = mat[6]
	m33 = mat[1]
	_z = math.asin(max(min(m21, 1), -1));

	if abs(m21) < 0.9999999:
		_x = math.atan2(-m23, m22);
		_y = math.atan2(-m31, m11);
	else:
		_x = 0;
		_y = math.atan2(m13, m33);

	return [_x, _y, _z]

if __name__ == "__main__":
	
	s = Sim()
	s.run()
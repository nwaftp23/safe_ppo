import numpy as np


def discounted_rewards(stopping_speed, stop_position, stuck_time, penalty, gamma, goal, percent_stop):
	'''Helper to find the discounted sum of returns,
	this will help me build my environment to induce
	the behavior that I want'''
	goal_from_leader = goal-240
	steps_to_stop = int(np.ceil(15/stopping_speed))
	opt_steps = int(np.ceil(goal_from_leader/15))
	opt = [-1]*opt_steps
	sum_opt = np.polyval(opt, gamma)
	opt_wait = [-1]*(opt_steps+how_long_to_stop)
	sum_opt_wait = np.polyval(opt_wait, gamma)
	opt_stuck = [-1]*(opt_steps+stuck_time+magic_number)
	sum_opt_stuck = np.polyval(opt_stuck, gamma)
	accident_location = int(np.ceil(stop_position/15))
	opt_penalty = [-1]*accident_location
	opt_penalty.insert(0,penalty)
	sum_penalty = np.polyval(opt_penalty, gamma)
	print('no stop risk neutral', sum_opt)
	print('no stop risk aware', sum_opt_wait)
	print('stop risk neutral', sum_penalty)
	print('stop risk aware', sum_opt_stuck)
	risk_aware = (1-percent_stop)*sum_opt_wait + percent_stop*sum_opt_stuck
	risk_neutral = (1-percent_stop)*sum_opt + percent_stop*sum_penalty 
	print('risk_aware sum of rewards', risk_aware)
	print('risk_neutral sum of rewards', risk_neutral)


discounted_rewards(990, 10, -100, 0.995, 1000,0.05)

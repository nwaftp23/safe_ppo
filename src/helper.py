import numpy as np


def discounted_rewards(cruising_speed, stopping_speed, stop_position, stuck_time, penalty, gamma, goal, percent_stop):
	'''Helper to find the discounted sum of returns,
	this will help me build my environment to induce
	the behavior that I want'''
	goal_from_leader = goal-240
	leader_at_goal = np.ceil(goal_from_leader/cruising_speed)*cruising_speed
	steps_to_stop = int(np.ceil(cruising_speed/stopping_speed))
	pl = np.array([cruising_speed]*steps_to_stop)
	pll = np.arange(steps_to_stop)*stopping_speed
	risk_averse_separation = np.sum(pl-pll)
	extra_steps = int(np.ceil((leader_at_goal-risk_averse_sepration)/cruising_speed)) 
	opt_steps = int(np.ceil(goal_from_leader/cruising_speed))
	opt = [-1]*opt_steps
	sum_opt = np.polyval(opt, gamma)
	opt_wait = [-1]*(opt_steps+extra_steps)
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

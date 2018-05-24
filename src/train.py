#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
from scipy.optimize import minimize_scalar
import os
import argparse
import signal
import Optimal_Stop
import pickle

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, obs_dim, act_dim


''' Add Augmented state space to then implement a similar scenario to
Yinlam Chow. Might or might not work. Other ideas not yet tested, leverage
our critic in our estimation of risk parameter, build a seperate risk net,
or increase the batch to ensure rare event encounter. Some methods can be
combined together.'''

'''

This section of code was commented out because it was the augmented MDP trial
Now I going with the simpler approach which should work
def run_episode(env, policy, scaler, init_var, gamma, animate = False, alpha, lambda_k):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        if step == 0:
            augie = init_var
        else:
            augie += (reward/gamma)
        obs = np.append(obs, [[augie ,step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
    new_var = minimize_scalar(f, args = (augie, init_var, lambda_k, alpha), bounds=(-2000000,2000000), method = 'bounded')
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs), new_var.x)

def f(x_new, augie, init_var, lambda_k, alpha):
    indicator = int(augie<=0)
    step_size3 = 0.01
    _ = init_var - step_size3*(lambda_k - lambda_k/(1-alpha)*indicator)
    return (x_new - _)**2



def run_policy(env, policy, scaler, logger, init_var, gamma, episodes, alpha, lambda_k):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs, new_var = run_episode(env, policy, scaler, init_var, gamma, alpha, lambda_k)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        init_var = new_var
        print('Var', new_var, 'at episode', e)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    rew = np.concatenate([t['rewards'] for t in trajectories])
    scaler.update(unscaled, rew)  # update running statistics for scaling observations
  # scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories , init_var
'''



def run_episode(env, policy, scaler, animate = False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))



def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    rew = np.concatenate([t['rewards'] for t in trajectories])
    scaler.update(unscaled, rew)  # update running statistics for scaling observations
  # scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories



def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma, mu, sig):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        #Ideas for new scaling, why using gamma. Just set
	    #a pre determined factor like 0.01, can tray a few
	    #also try a normalization like suggested in the paper
        '''Original Scaling'''
        #if gamma < 0.999:  # don't scale for gamma ~= 1
        #    rewards = trajectory['rewards'] * (1 - gamma)
        #else:
        #    rewards = trajectory['rewards']
        '''Standard deviation scaling'''
        rewards = normalize_rew(trajectory, mu, sig)
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

def normalize_rew(trajectory, mu, sig):
    if sig == 0:
        rewards = (trajectory['rewards'])
    else:
        rewards = (trajectory['rewards'])/np.sqrt(sig)
    return rewards


def add_disc_sum_rew_noscale(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories
    not scaled

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew_noscale'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam, mu, sig):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        # Lucas' correction
        # try reward scaling suggested in the paper
        '''Original Scaling'''
        #if gamma < 0.999:  # don't scale for gamma ~= 1
        #    rewards = trajectory['rewards'] * (1 - gamma)
        #else:
        #    rewards = trajectory['rewards']
        '''standard deviation scaling'''
        rewards = normalize_rew(trajectory, mu, sig)
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages

def VaR(disc_sum_rewards,threshold):
    """ Calculates the VaR of the discounted sum of returns.
    Takes as inputs discounted sum of rewards and alpha."""
    costs = -1*np.array(discounted_sum_rewards)
    Value_at_Risk = np.percentile(rewards,threshold)
    return Value_at_Risk

def CVaR(disc_sum_rewards,threshold):
    """ Calculates the CVaR of the discounted sum of returns.
    Takes as inputs discounted sum of rewards and alpha."""
    Value_at_Risk = VaR(disc_sum_rewards)
    tail_values = -1*np.array(list(filter(lambda a: a >= Value_at_Risk, e)))
    Conditional_Value_at_Risk = np.mean(tail_values)
    return Conditional_Value_at_Risk

def EVaR(rewards, threshold):
    """ Calculates the EVaR of the discounted sum of returns.
    Takes as inputs discounted sum of rewards and alpha.
    ----TODO-------
    Idea incorporate the natural gradient look at paper"""


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    #disc_sum_rew0 = np.array([t['disc_sum_rew_noscale'][0] for t in trajectories])
    return observes, actions, advantages, disc_sum_rew

def get_end_policy_dist(policy, n):
    run_policy(env, policy, scaler, logger, episodes=n)


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar, print_results, risk_targ):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now_utc = datetime.utcnow()  # create unique directories
    now = str(now_utc.day) + '-' + now_utc.strftime('%b') + '-' + str(now_utc.year) + '_' + str(((now_utc.hour-4)%24)) + '.' + str(now_utc.minute) + '.' + str(now_utc.second) # adjust for Montreal Time Zone
    logger = Logger(logname=env_name, now = now)
    aigym_path = os.path.join('/tmp', env_name, now)
    #env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, risk_targ,'CVaR', batch_size, 1)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    kl_terms = np.array([])
    beta_terms = np.array([])
    if print_results:
        rew_graph = np.array([])
        mean_rew_graph = np.array([])
    #big_li_rew_nodisc0 = np.array([])
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        #predicted_values_0 = [t['values'][0] for t in trajectories]
        add_disc_sum_rew(trajectories, gamma, scaler.mean_rew, np.sqrt(scaler.var_rew))  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam, scaler.mean_rew, np.sqrt(scaler.var_rew))  # calculate advantage
        nodisc0 = -1*np.array([t['rewards'].sum() for t in trajectories])
        disc0 = [t['disc_sum_rew'][0] for t in trajectories]
        #### WINDOW ####
        #rew_nodisc0 = [np.sum(t['rewards']) for t in trajectories]
        #big_li_rew_nodisc0= np.append(big_li_rew_nodisc0,rew_nodisc0)
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        lamb = policy.update(observes, actions, advantages, nodisc0, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        kl_terms = np.append(kl_terms,policy.check_kl)
        x1 = list(range(1,(len(kl_terms)+1)))
        rewards = plt.plot(x1,kl_terms)
        plt.title('RAPPO')
        plt.xlabel("Episode")
        plt.ylabel("KL Divergence")
        plt.savefig("KL_curve.png")
        plt.close()
        beta_terms = np.append(beta_terms,policy.beta)
        x2 = list(range(1,(len(beta_terms)+1)))
        mean_rewards = plt.plot(x2,beta_terms)
        plt.title('RAPPO')
        plt.xlabel("Batch")
        plt.ylabel("Beta Lagrange Multiplier")
        plt.savefig("lagrange_beta_curve.png")
        plt.close()
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
        if print_results:
            rew_graph = np.append(rew_graph,disc0)
            x1 = list(range(1,(len(rew_graph)+1)))
            rewards = plt.plot(x1,rew_graph)
            plt.title('RAPPO')
            plt.xlabel("Episode")
            plt.ylabel("Discounted sum of rewards")
            plt.savefig("learning_curve.png")
            plt.close()
            mean_rew_graph = np.append(mean_rew_graph,np.mean(disc0))
            x2 = list(range(1,(len(mean_rew_graph)+1)))
            mean_rewards = plt.plot(x2,mean_rew_graph)
            plt.title('RAPPO')
            plt.xlabel("Batch")
            plt.ylabel("Mean of Last Batch")
            plt.savefig("learning_curve2.png")
            plt.close()
    if print_results:
        tr = run_policy(env, policy, scaler, logger, episodes=1000)
        sum_rewww = [t['rewards'].sum() for t in tr]
        hist_dat = np.array(sum_rewww)
        fig = plt.hist(hist_dat,bins=2000, edgecolor='b', linewidth=1.2)
        plt.title('RAPPO')
        plt.xlabel("Sum of Rewards")
        plt.ylabel("Frequency")
        plt.savefig("RA_ppo.png")
        plt.close()
        with open('sum_rew_final_policy.pkl', 'wb') as f:
            pickle.dump(sum_rewww, f)
        logger.final_log()
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-e','--env_name', type=str, help='OpenAI Gym environment name', default = 'OptimalStop-v0')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=2000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.9995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-r', '--risk_targ', type=float, help='Risk target value or Constraint',
                        default = 400)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    #parser.add_argument('-vi', '--visualize', type=bool,
    #                    help='Visualize the training (needs to off for sshing).',
    #                    default=False)
    parser.add_argument('-pr', '--print_results', type=bool,
        help='Plot histogram of final policy',
                        default=False)

    args = parser.parse_args()
    main(**vars(args))

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
            augie = VaR
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
    new_VaR = minimize_scalar(f, args = (augie, init_var, lambda_k, alpha), bounds=(-2000000,2000000), method = 'bounded')
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


#### WINDOW ####
#rew_nodisc0 = [np.sum(t['rewards']) for t in trajectories]
#big_li_rew_nodisc0= np.append(big_li_rew_nodisc0,rew_nodisc0)
# concatenate all episodes into single NumPy arrays

#parser.add_argument('-vi', '--visualize', type=bool,
#                    help='Visualize the training (needs to off for sshing).',
#                    default=False)

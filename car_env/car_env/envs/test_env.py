"""from car_world import *
rolling = Safe_Stop()
#rolling.make_background('background2.jpeg')
rolling.rollout(.05)
"""

from Optimal_stop import *
done = False
env = Optimal_Stop()
env.open_pygame()
while not done:
    state, reward, done, _ = env.step(np.array([-.05]))
    env.render()

"""from car_world import *
rolling = Safe_Stop()
#rolling.make_background('background2.jpeg')
rolling.rollout(.05)
"""

from Optimal_stop import *

env = Optimal_Stop()
env.open_pygame()
for i in range(100):
    state, reward, done, driver_speed, _ = env.step(1)
    env.render(1, driver_speed)

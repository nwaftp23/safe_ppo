"""from car_world import *
rolling = Safe_Stop()
#rolling.make_background('background2.jpeg')
rolling.rollout(.05)
"""

from Optimal_Stop import *
done = False
env = Optimal_Stop()
for i in range(5):
    done = False
    env.reset()
    env.open_pygame()
    while not done:
        state, reward, done, _ = env.step(np.array([.005]))
        env.render()

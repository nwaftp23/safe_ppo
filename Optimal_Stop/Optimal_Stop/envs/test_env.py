"""from car_world import *
rolling = Safe_Stop()
#rolling.make_background('background2.jpeg')
rolling.rollout(.05)
"""

from Optimal_Stop import *
done = False
env = Optimal_Stop()
steppp = []
for i in range(1):
    done = False
    steps = 0
    env.reset()
    env.open_pygame()
    while not done:
        state, reward, done, _ = env.step(np.array([0.25]))
        env.render()
        steps += 1
    steppp.append(steps)
pygame.quit()

print('min steps', min(steppp))
print('max steps', max(steppp))
print('mean steps', np.mean(steppp))

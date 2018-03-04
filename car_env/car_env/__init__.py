from gym.envs.registration import register

register(
    id='car_env-v0',
    entry_point='car_env.envs:car_world',
)

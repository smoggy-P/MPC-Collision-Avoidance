from gym.envs.registration import register
register(
    id='Quadrotor-v0',
    entry_point='env.quadrotor1:Quadrotor',
)
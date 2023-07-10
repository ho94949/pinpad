from gym.envs.registration import register

register(
    id="gym_pinpad/PinPad-v0",
    entry_point="gym_pinpad.envs:PinPadEnv",
)
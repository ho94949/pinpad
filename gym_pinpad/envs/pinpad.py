import collections
import gym
import numpy as np
from . import settings


class PinPadEnv(gym.Env):
    r'''PinPad environment from Danijar - Director.'''

    def __init__(self, task='three', seed=0, length=1000):
        assert length > 0, f'length should be positive'
        layout = {
            'three': settings.LAYOUT_THREE,
            'four': settings.LAYOUT_FOUR,
            'five': settings.LAYOUT_FIVE,
            'six': settings.LAYOUT_SIX,
            'seven': settings.LAYOUT_SEVEN,
            'eight': settings.LAYOUT_EIGHT,
        }
        assert task in layout, f'Task: {task} not supported.'
        self.layout = np.array([list(ln) for ln in layout[task].split('\n')]).T
        assert self.layout.shape == (16, 12)

        self.length = length
        self.pads: set[str] = set(
            self.layout.flatten().tolist()) - set('* #\n')
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != '#':
                self.spawns.append((x, y))

        self.reset(seed=seed)

    def _respawn(self):
        target_pad = list(sorted(self.pads))
        self.random.shuffle(target_pad)
        self.target = tuple(target_pad)
        self.sequence = collections.deque(maxlen=len(self.target))
        self.player = self.spawns[self.random.randint(len(self.spawns))]

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.random = np.random.RandomState(self.seed)

        self._respawn()

        self.steps = 0
        self.countdown = 0
        self.done = False
        return self._obs()

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)

    @property
    def obs_space(self):
        spaces = {
            "observation": self._env.observation_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    def step(self, action):
        if self.done:
            return self.reset()
        if self.countdown:
            self.countdown -= 1
            if self.countdown == 0:
                self._respawn()
            else:
                return self._obs()

        reward = 0.0
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
        x = np.clip(self.player[0] + move[0], 0, self.layout.shape[0]-1)
        y = np.clip(self.player[1] + move[1], 0, self.layout.shape[1])

        # Check whether stepped on tile.
        tile = self.layout[x][y]
        if tile != '#':
            self.player = (x, y)
        if tile in self.pads:
            if not self.sequence or self.sequence[-1] != tile:
                self.sequence.append(tile)

        if tuple(self.sequence) == self.target and not self.countdown:
            reward += 1
            self.countdown = 10

        self.steps += 1
        self.done = self.done or (self.steps >= self.length)
        return self._obs(reward=reward, is_last=self.done)

    def render(self):
        grid = np.zeros((16, 16, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if self.countdown:
            # countdown background as light grey
            grid[:] = (223, 223, 223)

        current = self.layout[self.player[0]][self.player[1]]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)
            elif char in self.pads:
                color = np.array(settings.COLORS[char])
                # ambient coloring on stepped colour
                color = color if char == current else (color + 9 * white) / 10
                grid[x, y] = color

        grid[self.player] = (0, 0, 0)
        grid[:, -4:] = (192, 192, 192)
        for i, char in enumerate(self.target):
            grid[2*i+1, -4] = settings.COLORS[char]
        for i, char in enumerate(self.sequence):
            grid[2*i+1, -2] = settings.COLORS[char]

        return np.repeat(np.repeat(grid, 4, 0), 4, 1).transpose((1, 0, 2))

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _obs(self, reward=0.0, is_first=False, is_last=False, is_terminal=False):
        return dict(image=self.render(), reward=reward, is_first=is_first, is_last=is_last, is_terminal=is_terminal)

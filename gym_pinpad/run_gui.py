import argparse

import numpy as np
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise
from PIL import Image

import gym_pinpad.envs


def main():
    def boolean(x): return bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='three')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--window', type=int, nargs=2, default=(640, 640))
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--wait', type=boolean, default=False)
    parser.add_argument('--death', type=str, default='reset', choices=[
        'continue', 'reset', 'quit'])
    args = parser.parse_args()

    keymap = {
        pygame.K_a: 'move_left',
        pygame.K_d: 'move_right',
        pygame.K_w: 'move_up',
        pygame.K_s: 'move_down',
        pygame.K_SPACE: 'noop',
    }
    print('Actions:')
    for key, action in keymap.items():
        print(f'  {pygame.key.name(key)}: {action}')

    env = gym_pinpad.envs.PinPadEnv(
        task=args.task, seed=args.seed, length=args.length)
    prev_step = env.reset()

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()
    running = True
    total_reward = 0

    while running:
        image = prev_step['observation']
        image = Image.fromarray(image)
        image = image.resize(args.window, resample=Image.NEAREST)
        image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

        # key input
        action = None
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
                action = keymap[event.key]

        if action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                if args.wait:
                    continue
                else:
                    action = 'noop'

        prev_step = env.step(
            ['noop', 'move_down', 'move_up', 'move_right', 'move_left'].index(action))

        if prev_step['reward'] != 0:
            print(f'Reward: {prev_step["reward"]}')
            total_reward += prev_step['reward']
        if prev_step['is_last']:
            print('Episode Done!')
            print(f'Total Reward: {total_reward}')
            running = False
            pygame.quit()


if __name__ == '__main__':
    main()

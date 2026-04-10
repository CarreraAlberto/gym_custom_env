from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# This code is based on the example from Gymnasium:
# https://gymnasium.farama.org/introduction/create_custom_env/
#
# This environment implements a Coverage Path Planning (CPP) task.
# The agent (blue circle) must visit every accessible (non-obstacle) cell in the grid.
# There is no fixed goal location; the episode ends when full coverage is achieved
# or when the step budget is exhausted.
#
# The state is represented as a flattened array containing:
# - the agent's (x, y) location
# - the visited map (size x size), where 0 = not visited and 1 = visited
# - the state of the 4 neighboring cells (right, up, left, down),
#   where 0 indicates a free cell and 1 indicates an obstacle or wall.
#
# The action space is discrete with 4 actions: move right, up, left, down.
#
# The agent receives:
# - a reward of +1.0 for visiting a new (previously unvisited) cell,
# - a penalty of -0.5 for revisiting a cell already covered,
# - a small step cost of -0.1 applied every step,
# - a completion bonus of +10.0 when all accessible cells have been visited.
#
# The episode ends when the agent achieves full coverage or after a maximum number of steps.

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    R_NEW_CELL = 1.0
    R_REVISIT  = -0.5
    R_STEP     = -0.1
    R_COMPLETE = 10.0

    def __init__(self, render_mode=None, size: int = 5,
                 obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.count_steps = 0
        self.max_steps = max_steps

        self.obstacles_locations = []
        self.visited = np.zeros((size, size), dtype=int)

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.array([0, 0, 0, 0], dtype=int)  #right, up, left, down

        # The state is represented with the agent's location, the visited map and the grid of neighbors
        n_obs = 2 + size * size + 4
        low  = np.zeros(n_obs, dtype=int)
        high = np.array([size - 1, size - 1] + [1] * (size * size) + [1] * 4, dtype=int)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([ 1,  0]),  # right
            1: np.array([ 0, -1]),  # up
            2: np.array([-1,  0]),  # left
            3: np.array([ 0,  1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        flattened = []
        flattened.extend(self._agent_location)
        flattened.extend(self.visited.flatten())
        flattened.extend(self._neighbors)
        return np.array(flattened, dtype=int)

    def _get_info(self):
        accessible = self.size * self.size - len(self.obstacles_locations)
        covered = int(self.visited.sum())
        return {
            "coverage_ratio": covered / accessible if accessible > 0 else 1.0,
            "covered_cells": covered,
            "accessible_cells": accessible,
            "steps": self.count_steps,
        }

    def set_neighbors(self, obstacles_locations):
        # create a map of the neighbors
        # 1 = free, 0 = obstacle or wall
        directions = [np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])]
        for i, direction in enumerate(directions):
            neighbor = self._agent_location + direction
            if (0 <= neighbor[0] < self.size) and (0 <= neighbor[1] < self.size) and not any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                self._neighbors[i] = 0
            else:
                self._neighbors[i] = 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = np.zeros((self.size, self.size), dtype=int)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location
            while (np.array_equal(obstacle_location, self._agent_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        # Mark the starting cell as visited
        x, y = self._agent_location
        self.visited[x, y] = 1

        self.set_neighbors(self.obstacles_locations)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # If the agent hits an obstacle, it stays in the same position
        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        x, y = self._agent_location
        new_cell = (self.visited[x, y] == 0)
        self.visited[x, y] = 1

        accessible = self.size * self.size - len(self.obstacles_locations)
        covered = int(self.visited.sum())
        terminated = (covered == accessible)
        truncated = (not terminated) and (self.count_steps >= self.max_steps)

        # An environment is completed if and only if the agent has covered all accessible cells
        if new_cell:
            reward = self.R_NEW_CELL + self.R_STEP
        else:
            reward = self.R_REVISIT + self.R_STEP

        if terminated:
            reward += self.R_COMPLETE

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("Coverage Path Planning")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw visited cells (light green)
        for xi in range(self.size):
            for yi in range(self.size):
                if self.visited[xi, yi] == 1:
                    pygame.draw.rect(
                        canvas,
                        (144, 238, 144),
                        pygame.Rect(
                            pix_square_size * xi,
                            pix_square_size * yi,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )

        # Draw the obstacles
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Coverage text overlay
        if self.render_mode == "human":
            font = pygame.font.SysFont(None, 28)
            accessible = self.size * self.size - len(self.obstacles_locations)
            covered = int(self.visited.sum())
            label = font.render(
                f"Coverage: {covered}/{accessible} ({100 * covered // accessible}%)",
                True, (60, 60, 60),
            )
            canvas.blit(label, (4, 4))

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

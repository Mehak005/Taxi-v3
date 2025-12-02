import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TaxiLargeEnv(gym.Env):

    metadata = {"render_modes": ["ansi"], "render_fps": 5}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # (taxi_row, taxi_col, passenger_loc, destination)
        self.observation_space = spaces.Discrete(
            grid_size * grid_size * 5 * 4
        )

        # Actions identical to Taxi-v3
        # 0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff
        self.action_space = spaces.Discrete(6)

        # Locations (expanded)
        self.locs = [
            (0, 0),
            (0, grid_size-1),
            (grid_size-1, 0),
            (grid_size-1, grid_size-1)
        ]

        # Passenger can be at 4 locations or in the taxi (location = 4)
        self.in_taxi = 4

        # Build walls scaled up from original Taxi layout
        self._build_walls()

    def _build_walls(self):
        """Scale original Taxi walls to larger grids."""
        self.walls = set()

        # Original Taxi left wall between (0,0)-(1,0) becomes scaled vertical walls
        scale = self.grid_size // 5
        for i in range(self.grid_size):
            if (i // scale) in [1, 2]:  # mimic original structure
                self.walls.add((i, self.grid_size // 2))  # vertical wall

    def encode(self, taxi_row, taxi_col, passenger, dest):
        """Encode state into 1 integer (same as Taxi-v3 logic)."""
        i = taxi_row
        i *= self.grid_size
        i += taxi_col
        i *= 5
        i += passenger
        i *= 4
        i += dest
        return i

    def decode(self, i):
        """Decode integer state into components."""
        out = []

        dest = i % 4
        i //= 4
        out.append(dest)

        passenger = i % 5
        i //= 5
        out.append(passenger)

        col = i % self.grid_size
        i //= self.grid_size
        out.append(col)

        row = i
        out.append(row)

        taxi_row = row
        taxi_col = col
        passenger = out[1]
        dest = out[0]
        return taxi_row, taxi_col, passenger, dest

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.taxi_row = self.np_random.integers(self.grid_size)
        self.taxi_col = self.np_random.integers(self.grid_size)
        self.passenger_loc = self.np_random.integers(5)   # includes in_taxi=4
        self.dest = self.np_random.integers(4)

        state = self.encode(
            self.taxi_row, self.taxi_col, self.passenger_loc, self.dest
        )
        return state, {}

    def is_wall(self, row, col, new_row, new_col):
        """Blocks movement if a wall exists between two cells."""
        # Vertical walls only (like original Taxi)
        if (row, col) in self.walls and (new_row == row and new_col == col + 1):
            return True
        if (row, col - 1) in self.walls and (new_row == row and new_col == col - 1):
            return True
        return False

    def step(self, action):
        reward = -1

        new_row, new_col = self.taxi_row, self.taxi_col

        if action == 0 and self.taxi_row < self.grid_size - 1:  # South
            new_row = self.taxi_row + 1
        elif action == 1 and self.taxi_row > 0:  # North
            new_row = self.taxi_row - 1
        elif action == 2 and self.taxi_col < self.grid_size - 1:  # East
            new_col = self.taxi_col + 1
        elif action == 3 and self.taxi_col > 0:  # West
            new_col = self.taxi_col - 1

        # Block if a wall is in the way
        if not self.is_wall(self.taxi_row, self.taxi_col, new_row, new_col):
            self.taxi_row, self.taxi_col = new_row, new_col

        # Pickup
        if action == 4:
            if self.passenger_loc != self.in_taxi:
                px, py = self.locs[self.passenger_loc]
                if (self.taxi_row, self.taxi_col) == (px, py):
                    self.passenger_loc = self.in_taxi
                else:
                    reward = -10

        # Dropoff
        if action == 5:
            if self.passenger_loc == self.in_taxi:
                dx, dy = self.locs[self.dest]
                if (self.taxi_row, self.taxi_col) == (dx, dy):
                    reward = 20
                    terminated = True
                else:
                    reward = -10
            else:
                reward = -10

        state = self.encode(
            self.taxi_row, self.taxi_col, self.passenger_loc, self.dest
        )

        terminated = reward == 20
        truncated = False

        return state, reward, terminated, truncated, {}

    def render(self):
        """Simple ASCII rendering."""
        out = ""
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                if (r, c) == (self.taxi_row, self.taxi_col):
                    row += "T "
                else:
                    row += ". "
            out += row + "\n"
        return out

from gymnasium.envs.registration import register

register(
    id="TaxiLarge-v3",
    entry_point="taxi_large_env:TaxiLargeEnv",
    kwargs={"grid_size": 10},
)


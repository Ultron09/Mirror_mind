#!/usr/bin/env python3
"""
Protocol V4 - Practical multi-domain agent protocol
--------------------------------------------------

This module implements `ProtocolV4`, a practical protocol scaffold
designed for real-life usage across several agent types:
- Path-finder CNN-based robot (navigation)
- LLM inference provider (serving predictions)
- Robot controller (actuation loop)
- Virtual employee lifecycle (intern -> employee)

The design focuses on modular components that can be composed into
real systems. The module includes a small demo that runs synthetic
scenarios and emits a JSON report `protocol_v4_report.json`.

Math and important formulas are explained in the accompanying
`protocol_v4_report.md` file.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import time
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Perception: Small CNN for pathfinder
# ------------------------------
class PerceptionCNN(nn.Module):
    """Small convolutional network that maps a top-down grid to logits.

    Input: tensor [B, C=1, H, W]
    Output: tensor [B, num_actions]
    """
    def __init__(self, num_actions: int = 5, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)


# ------------------------------
# Planner: simple grid A* placeholder
# ------------------------------
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """Very small A* pathfinder for a binary occupancy grid.

    grid: 0 free, 1 obstacle
    Returns a list of coordinates representing the path.
    """
    w = len(grid[0])
    h = len(grid)
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    def neighbors(p):
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == 0:
                yield (nx, ny)

    while open_set:
        current = min(open_set, key=lambda o: f_score.get(o, 1e9))
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        open_set.remove(current)
        for n in neighbors(current):
            tentative_g = g_score.get(current, 1e9) + 1
            if tentative_g < g_score.get(n, 1e9):
                came_from[n] = current
                g_score[n] = tentative_g
                f_score[n] = tentative_g + heuristic(n, goal)
                open_set.add(n)
    return []


# ------------------------------
# LLM Provider Simulator
# ------------------------------
class LLMProvider:
    """Simulates an LLM inference provider with latency and token costing.

    This is a simplified stand-in for a real LLM service.
    """
    def __init__(self, base_latency_ms: float = 20.0, tokens_per_call: int = 32):
        self.base_latency_ms = base_latency_ms
        self.tokens_per_call = tokens_per_call

    def infer(self, prompt: str) -> Dict[str, Any]:
        # Simulated latency
        latency = random.gauss(self.base_latency_ms, self.base_latency_ms * 0.1)
        time.sleep(max(0.0, latency / 1000.0))
        # Fake response
        tokens = self.tokens_per_call
        cost = tokens * 0.00001
        return {
            'response': prompt[::-1][:256],  # deterministic fake reply: reversed prompt
            'latency_ms': latency,
            'tokens': tokens,
            'cost_usd': cost
        }


# ------------------------------
# Robot Controller (simulation)
# ------------------------------
class RobotController:
    """Simple kinematic simulation for a point robot in grid world."""
    def __init__(self, start: Tuple[int, int] = (0, 0)):
        self.pos = start

    def step(self, action: Tuple[int, int]) -> Tuple[int, int]:
        # action is delta (dx, dy) clamped to grid step
        x, y = self.pos
        dx, dy = action
        self.pos = (x + int(math.copysign(1, dx)) if dx != 0 else x,
                    y + int(math.copysign(1, dy)) if dy != 0 else y)
        return self.pos


# ------------------------------
# Virtual Employee Lifecycle
# ------------------------------
@dataclass
class VirtualEmployee:
    name: str
    skill: float = 0.0
    experience: float = 0.0
    level: str = 'intern'  # intern -> junior -> mid -> senior -> employee

    def update(self, reward: float, learning_rate: float = 0.1):
        """Update skill and experience based on reward signal.

        Simple rule: skill <- skill + lr * reward, experience increases
        Level thresholds are deterministic for demo.
        """
        self.skill += learning_rate * reward
        self.experience += abs(reward)
        self._recalculate_level()

    def _recalculate_level(self):
        if self.experience < 10:
            self.level = 'intern'
        elif self.experience < 30:
            self.level = 'junior'
        elif self.experience < 80:
            self.level = 'mid'
        else:
            self.level = 'senior'


# ------------------------------
# EWC penalty function (math)
# ------------------------------
def ewc_penalty(theta: List[torch.Tensor], theta_star: List[torch.Tensor], fisher: List[torch.Tensor], lam: float = 1.0) -> torch.Tensor:
    """Compute EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    theta, theta_star, fisher are lists of tensors matching parameter shapes.
    """
    loss = 0.0
    for t, ts, f in zip(theta, theta_star, fisher):
        loss = loss + (f * (t - ts).pow(2)).sum()
    return loss * (lam / 2.0)


# ------------------------------
# ProtocolV4 orchestrator
# ------------------------------
@dataclass
class ProtocolV4:
    mode: str = 'virtual_employee'  # 'pathfinder', 'llm_provider', 'robot', 'virtual_employee'
    device: str = 'cpu'
    perception: Optional[PerceptionCNN] = None
    llm: Optional[LLMProvider] = None
    controller: Optional[RobotController] = None
    report: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode == 'pathfinder':
            self.perception = PerceptionCNN(num_actions=5)
        if self.mode == 'llm_provider':
            self.llm = LLMProvider()
        if self.mode == 'robot':
            self.controller = RobotController()
        if self.mode == 'virtual_employee':
            self.employee = VirtualEmployee(name='Alex')

    # ---- Pathfinding demo ----
    def run_pathfinder_episode(self, grid_size: int = 16) -> Dict[str, Any]:
        # Create random occupancy grid
        grid = [[0 if random.random() > 0.2 else 1 for _ in range(grid_size)] for __ in range(grid_size)]
        start = (0, 0)
        goal = (grid_size - 1, grid_size - 1)
        # Ensure start/goal free
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0

        path = a_star(start, goal, grid)
        success = len(path) > 0
        self.report['pathfinder'] = {'grid_size': grid_size, 'path_found': success, 'path_length': len(path)}
        return self.report['pathfinder']

    # ---- LLM provider demo ----
    def run_llm_inference(self, prompt: str) -> Dict[str, Any]:
        assert self.llm is not None, 'LLM provider not initialized'
        result = self.llm.infer(prompt)
        self.report['llm'] = result
        return result

    # ---- Robot simulation demo ----
    def run_robot_sim(self, steps: int = 10) -> Dict[str, Any]:
        assert self.controller is not None, 'Robot controller not initialized'
        traj = [self.controller.pos]
        for _ in range(steps):
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            pos = self.controller.step((dx, dy))
            traj.append(pos)
        self.report['robot'] = {'trajectory': traj}
        return self.report['robot']

    # ---- Virtual employee lifecycle demo ----
    def run_virtual_employee_training(self, episodes: int = 20) -> Dict[str, Any]:
        # Simulate episodic tasks with reward signals
        for e in range(episodes):
            # reward distribution depends on level; interns start low
            reward = random.gauss(0.5, 0.5) + (0.05 * self.employee.experience)
            self.employee.update(reward=max(-1.0, min(1.0, reward)))

        self.report['virtual_employee'] = {
            'name': self.employee.name,
            'skill': self.employee.skill,
            'experience': self.employee.experience,
            'level': self.employee.level,
        }
        return self.report['virtual_employee']

    # ---- Utility: produce final report file ----
    def dump_report(self, filename: str = 'protocol_v4_report.json') -> str:
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2)
        return filename


# ------------------------------
# Demo main
# ------------------------------
def main():
    # Run a short demo for all modes and collect results
    proto = ProtocolV4(mode='virtual_employee')
    proto.run_virtual_employee_training(episodes=25)
    proto.dump_report('protocol_v4_virtual_employee.json')

    proto_pf = ProtocolV4(mode='pathfinder')
    proto_pf.run_pathfinder_episode(grid_size=16)
    proto_pf.dump_report('protocol_v4_pathfinder.json')

    proto_llm = ProtocolV4(mode='llm_provider')
    proto_llm.run_llm_inference('Hello MirrorMind, summarize protocol v4')
    proto_llm.dump_report('protocol_v4_llm.json')

    proto_robot = ProtocolV4(mode='robot')
    proto_robot.run_robot_sim(steps=12)
    proto_robot.dump_report('protocol_v4_robot.json')

    # Combined report
    combined = {
        'virtual_employee': proto.report.get('virtual_employee'),
        'pathfinder': proto_pf.report.get('pathfinder'),
        'llm': proto_llm.report.get('llm'),
        'robot': proto_robot.report.get('robot'),
        'note': 'All outputs are synthetic demo values; replace with real datasets and models for production.'
    }
    with open('protocol_v4_report.json', 'w') as f:
        json.dump(combined, f, indent=2)

    print('\nProtocol V4 demo complete. Reports written: protocol_v4_report.json')


if __name__ == '__main__':
    main()

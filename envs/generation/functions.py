"""This file describes all functions which can be used as cause-effect functions in an SCM"""
from typing import List
import random


def f_linear_noise(parents: List[str]):
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    default_value = 0.0

    def f(**kwargs):
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu + random.gauss(0, 0.5)

    return f


def f_linear(parents: List[str]):
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    default_value = 0.0

    def f(**kwargs):
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu

    return f


def f_interaction(parents: List[str]):
    if len(parents) != 0:
        interact_par1 = random.choice(parents)
        interact_par2 = random.choice(parents)

    def interaction(**kwargs):
        mu = 0
        for p in parents:
            mu += kwargs[p]
        if len(parents) != 0:
            mu += kwargs[interact_par1] * kwargs[interact_par2]
        return mu
    return interaction


import numpy as np
import voxart

def test_create_design_random():
    rng = np.random.default_rng()
    def random_goal():
        return rng.choice(a=[0, 1], p=[0.7, 0.3], size=(7, 7))
    goal = voxart.Goal.from_arrays(
        random_goal(), random_goal(), random_goal())
    design = voxart.create_design_random(goal)

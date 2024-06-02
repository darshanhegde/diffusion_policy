import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from diffusion_policy.env.pusht.pusht_env import PushTEnv
import cv2
import collections
import numpy as np

env = PushTEnv(A = 1, b = 225, shade_above = False)
env.seed(100000)

# get first observation
env.reset()
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

imgs = np.array(imgs).astype(np.uint8)

while True:
    cv2.imshow('Default Env Image', imgs[0])
    # Check for window close event or key press
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    if cv2.getWindowProperty('Default Env Image', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

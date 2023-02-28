# Register dataset classes here
from .replay_buffer import ReplayBuffer, HindsightReplayBuffer
from .antmaze_dataset import GoalConditionedAntDataset
from .wgcsl_dataset import WGCSLDataset
from .kitchen_dataset import KitchenDataset
from .bridge_dataset import BridgeDataset

try:
    from .rmimic_dataset import GoalConditionedRobomimicDataset
except:
    print("[research] Failed to import robomimic dataset.")

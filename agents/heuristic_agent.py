from typing import Any, Dict, Optional, Union

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class HeuristicAgent(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        raw_state = batch.info['raw_state']  # List of state pro env
        mask = batch.obs.mask
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        return Batch(act=logits.argmax(axis=-1))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}

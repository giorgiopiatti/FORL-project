from typing import Any
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from tianshou.data import (
    HERReplayBuffer,
    HERReplayBufferManager,
    PrioritizedReplayBuffer,
    PrioritizedReplayBufferManager,
    ReplayBuffer,
    ReplayBufferManager,
)

from numba import njit

from tianshou.data import Batch, HERReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
from tianshou.data.batch import _alloc_by_keys_diff, _create_value


class MultiAgentVectorReplayBuffer(ReplayBufferManager):
    """VectorReplayBuffer contains n ReplayBuffer with the same size.

    It is used for storing transition from different environments yet keeping the order
    of time.

    :param int total_size: the total size of VectorReplayBuffer.
    :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
        configuration.

    Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
    are the same as :class:`~tianshou.data.ReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # preprocess batch
        if self.__len__() > 0:
            unfinished_buffers = []   
            
            last_index = []
            for i in buffer_ids:
                unfinished_buffers.append(len(self.buffers[i].unfinished_index()) > 0)
                last_index.append(self.buffers[i].unfinished_index() + self._offset[i])

            unfinished_buffers = np.array(unfinished_buffers)        
            last_index = np.concatenate(last_index)

            if unfinished_buffers.any(): # Episode is not finished, update obs_next of last entry
                self._meta.obs_next[last_index] = batch.obs[unfinished_buffers]
        
        return super().add(batch, buffer_ids)

    def update_obs_next_end_episode(self,
        batch: Batch,
        reward: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
        ):

        last_index = []
        for i in buffer_ids:
            last_index.append([(self.buffers[i]._index - 1) % self.buffers[i]._size + self._offset[i]])
        last_index = np.concatenate(last_index)
        
        self._meta.obs_next[last_index] = batch
        self._meta.terminated[last_index] = np.ones_like(last_index, dtype=np.bool_)
        self._meta.done[last_index] =  np.ones_like(last_index, dtype=np.bool_)
        self._meta.rew[last_index] = reward

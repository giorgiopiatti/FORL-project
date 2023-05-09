
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from tianshou.env.utils import gym_new_venv_step_type
from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.utils import RunningMeanStd
from tianshou.env.venv_wrappers import VectorEnvWrapper


class VectorEnvNormObs(VectorEnvWrapper):
    """An observation normalization wrapper for vectorized environments.

    :param bool update_obs_rms: whether to update obs_rms. Default to True.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
    ) -> None:
        super().__init__(venv)
        # initialize observation running mean/std
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        obs, info = self.venv.reset(id, **kwargs)

        raw_obs = np.concatenate(list(map(lambda x: [x['obs']], obs)))
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(raw_obs)
        raw_obs = self._norm_obs(raw_obs)

        for i, o in enumerate(obs):
            o['obs'] = raw_obs[i]

        return obs, info

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> gym_new_venv_step_type:
        step_results = self.venv.step(action, id)

        raw_obs = np.concatenate(
            list(map(lambda x: [x['obs']], step_results[0])))

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(raw_obs)

        raw_obs = self._norm_obs(raw_obs)
        for i, o in enumerate(step_results[0]):
            o['obs'] = raw_obs[i]

        return (step_results[0], *step_results[1:])

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            return self.obs_rms.norm(obs)  # type: ignore
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms

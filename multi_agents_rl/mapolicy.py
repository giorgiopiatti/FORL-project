from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class MultiAgentPolicyManager(BasePolicy):
    """Multi-agent policy manager for MARL.

    This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(
        self, policies: List[BasePolicy], env: PettingZooEnv, learning_policies: List[str], **kwargs: Any
    ) -> None:
        super().__init__(action_space=env.action_space, **kwargs)
        assert (
            len(policies) == len(env.agents)
        ), "One policy must be assigned for each agent."
        self.learning_policies = learning_policies
        self.agent_idx = env.agent_idx
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(env.agents[i])

        self.policies = dict(zip(env.agents, policies))

    def replace_policy(self, policy: BasePolicy, agent_id: int) -> None:
        """Replace the "agent_id"th policy in this manager."""
        policy.set_agent_id(agent_id)
        self.policies[agent_id] = policy

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = policy.exploration_noise(
                act[agent_index], batch[agent_index]
            )
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's forward.

        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": actions corresponding to the input
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results: List[Tuple[bool, np.ndarray, Batch, Union[np.ndarray, Batch],
                            Batch]] = []
        for agent_id, policy in self.policies.items():
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2] (with batch_size == 6)
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent according to agent_id
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append(
                    (False, np.array([-1]), Batch(), Batch(), Batch()))
                continue
            tmp_batch = batch[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent_id]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            out = policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs
            )
            act = out.act
            each_state = out.state \
                if (hasattr(out, "state") and out.state is not None) \
                else Batch()
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat(
            [
                {
                    "act": act
                } for (has_data, agent_index, out, act, each_state) in results
                if has_data
            ]
        )
        state_dict, out_dict = {}, {}
        for (agent_id, _), (has_data, agent_index, out, act,
                            state) in zip(self.policies.items(), results):
            if has_data:
                holder.act[agent_index] = act
            state_dict[agent_id] = state
            out_dict[agent_id] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        raise NotImplementedError()

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}
        self.updating = True

        results = {}
        for agent_id, policy in self.policies.items():
            if agent_id not in self.learning_policies:
                continue
            batch, indices = buffer[agent_id].sample(sample_size)
            batch = self.process_fn(batch, buffer[agent_id], indices)
            data = batch[agent_id]
            if not data.is_empty():
                out = policy.learn(batch=data, **kwargs)
                for k, v in out.items():
                    results[agent_id + "/" + k] = v
            self.post_process_fn(batch, buffer[agent_id], indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return results
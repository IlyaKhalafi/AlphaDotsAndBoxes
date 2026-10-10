"""One reusable CUDA graph for padded self-play batches; training extra only."""

import torch
from ray.rllib.core.columns import Columns

from alphaboxes.game import State
from alphaboxes.graph import encode_batch


class CudaInference:
    def __init__(self, module, batch_size: int, capacity: int):
        self.module = module
        self.batch_size, self.capacity = batch_size, capacity
        self.graph = None

    def _initialize(self, state: State):
        device = next(self.module.parameters()).device
        self.observations = {
            key: torch.from_numpy(value).to(device)
            for key, value in encode_batch([state] * self.batch_size, self.capacity).items()
        }
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.module.predict(self.observations)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            result = self.module.predict(self.observations)
            self.probabilities = result[Columns.ACTION_DIST_INPUTS].softmax(dim=-1)
            self.values = result[Columns.VF_PREDS]

    def predict(self, states: list[State]):
        if not states:
            return []
        if len(states) > self.batch_size:
            raise ValueError("Prediction batch exceeds the CUDA graph capacity.")
        if self.graph is None:
            self._initialize(states[0])
        observations = encode_batch(states, self.capacity)
        for key, value in observations.items():
            self.observations[key][: len(states)].copy_(torch.from_numpy(value))
        self.graph.replay()
        probabilities = self.probabilities[: len(states)].cpu().numpy()
        values = self.values[: len(states)].cpu().numpy()
        return [
            (probabilities[i, : state.board.num_edges], float(values[i]))
            for i, state in enumerate(states)
        ]

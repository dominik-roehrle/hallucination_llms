import transformers
import torch 

class TokenStoppingCriteria(transformers.StoppingCriteria):
    """Stopping criteria that stops generation when a certain number of ### tokens are generated"""
    def __init__(self, sentinel_token_ids: torch.Tensor, 
                 starting_idx: int, counter: int, stop_counter: int):
        super().__init__()
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.counter = counter
        self.stop_counter = stop_counter

    def __call__(self, input_ids: torch.Tensor, _scores: torch.Tensor) -> bool:
        self.counter = 0
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            if trimmed_sample.shape[0] < 1:
                continue

            for token_id in trimmed_sample:
                if token_id in self.sentinel_token_ids:
                    if self.counter == self.stop_counter:
                        return True
                    else:
                        self.counter += 1
        return False
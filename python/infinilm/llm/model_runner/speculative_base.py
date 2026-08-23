"""Shared machinery for speculative-decoding runners.

Both the Eagle draft-model runner and the MiMo MTP runner (draft via the
target model's own MTP head) share: greedy forward wrappers, accept/total
counters, and the greedy acceptance matcher. Cache handling differs per
backend and lives in each subclass.
"""


class SpeculativeRunnerBase:
    """Base class for speculative-decoding runners.

    Subclasses implement `forward(scheduler_output, model_input)` and their own
    draft generation; they reuse `_greedy_forward_raw` (sampling forced greedy)
    and `_match_drafts` (accept the longest prefix of drafted tokens that the
    target model confirms).
    """

    def __init__(self, config, target_model_engine):
        self.config = config
        self.target_model_engine = target_model_engine
        self.num_draft_tokens = config.num_draft_tokens
        self.target_accept_count = 0
        self.target_total_count = 0

    def _greedy_forward_raw(self, inputs: dict) -> dict:
        """Run the target model forward with greedy sampling."""
        inputs = dict(inputs)
        inputs.update(temperature=1.0, top_k=1, top_p=1.0)
        return self.target_model_engine.forward_raw(**inputs)

    def _match_drafts(self, segment: list, draft_tokens: list):
        """Match drafted tokens against the target model's sampled tokens.

        `draft_tokens[0]` is the token at the current frontier; the target
        model's verification samples `segment[i]` at the position of
        `draft_tokens[i]` (predicting `draft_tokens[i+1]`). `segment` and
        `draft_tokens` cover the same verification window and are equal in
        length. We accept the longest prefix of `draft_tokens` that the target
        confirms, then take the target's next proposal as the correction.

        Returns `(accepted, correction)` with `accepted >= 1`.
        """
        accepted = 1
        correction = None
        for draft_idx in range(1, len(draft_tokens)):
            expected = int(segment[draft_idx - 1])
            if draft_tokens[draft_idx] != expected:
                correction = expected
                break
            accepted += 1
        if correction is None:
            correction = int(segment[len(draft_tokens) - 1])
        return accepted, correction

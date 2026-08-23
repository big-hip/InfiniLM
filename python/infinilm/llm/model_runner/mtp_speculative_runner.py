"""MiMo MTP speculative decoding.

The MiMo model embeds a Multi-Token Prediction head (`model.mtp_layers.N`)
alongside its Qwen2 backbone. At generation time the MTP head acts as a
self-draft: given the backbone's last hidden state and the token it just
proposed, it predicts the next-next token, and its own output chains to the
next draft. The target model then verifies the drafted tokens in one forward
and we accept the longest matching prefix (exact greedy speculation).

Unlike the Eagle runner (a separate draft model with its own paged KV), the
MTP head lives inside the target engine, so the draft is produced by the same
engine via `Input::target_hidden_states` (which `MiMoForCausalLM::forward`
dispatches to the MTP path). It uses the MTP KV cache slot that
`default_allocate_kv_cache_tensors` reserves past the backbone layers.

Scope: single request (batch=1) and static KV cache — the MTP cache slot is
shared and the static attention backend is batch=1. Multi-request MTP is a
later extension.
"""

import infinicore

from .speculative_base import SpeculativeRunnerBase


class MtpSpeculativeRunner(SpeculativeRunnerBase):
    """Speculative decoding using MiMo's MTP head as the drafter."""

    def __init__(self, config, target_model_engine, device):
        super().__init__(config, target_model_engine)

    def forward(self, scheduler_output, model_input):
        # Exact speculative acceptance is only valid for greedy sampling; keep
        # non-greedy requests on the plain target path.
        if self.config.top_k != 1 or self.config.temperature != 1.0:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []
        if len(requests) != 1:
            raise RuntimeError(
                "MTP speculative decoding currently requires batch=1 (static KV "
                f"cache); got {len(requests)} requests."
            )
        req = requests[0]

        target_output = self._greedy_forward_raw(model_input)
        target_token_ids = target_output["output_ids"].to_numpy().tolist()
        if not target_token_ids:
            return []
        hidden_states = target_output["hidden_states"]

        input_offsets = model_input["input_offsets"].to_numpy().tolist()
        last_input_idx = int(input_offsets[1]) - 1
        target_token = int(target_token_ids[last_input_idx])

        max_tokens = req.sampling_params.max_tokens
        remaining = (
            None
            if max_tokens is None
            else max_tokens - req.get_num_generated_tokens()
        )
        if remaining is not None and remaining <= 1:
            return [[target_token]]

        base_len = req.get_total_length()
        # Guard against the static cache overflowing during verification.
        draft_budget = min(self.num_draft_tokens, self.config.max_cache_len - base_len)
        if remaining is not None:
            draft_budget = min(draft_budget, max(1, remaining - 1))
        if draft_budget <= 0:
            return [[target_token]]

        # On the first forward, populate the MTP KV cache over the prompt so the
        # draft chain below attends to the true prefix.
        if scheduler_output.is_prefill and base_len > 1:
            self._seed_mtp_cache(req, hidden_states, base_len)

        # `combined` is the full speculative window: the token the backbone just
        # proposed (`target_token`, at `base_len - 1`) plus the MTP drafts for
        # the following positions. The target model verifies them all at once.
        chain_budget = draft_budget - 1
        mtp_drafts = self._draft_mtp_tokens(
            target_token, hidden_states, last_input_idx, base_len, chain_budget
        )
        combined = [target_token] + mtp_drafts

        segment = self._static_verify(combined, base_len)
        accepted, correction = self._match_drafts(segment, combined)
        self.target_total_count += len(combined) - 1
        self.target_accept_count += accepted - 1

        output_tokens = combined[:accepted] + [correction]
        if remaining is not None:
            output_tokens = output_tokens[:remaining]
        return [output_tokens]

    def _seed_mtp_cache(self, req, hidden_states, base_len):
        """Fill the MTP KV cache for prompt positions [0, base_len-2].

        The MTP layer at position `p` consumes `[h[p] | embed[x_{p+1}]]`; the
        seed runs it over the true prompt so the draft chain below attends to
        the real prefix.
        """
        prompt = list(req.prompt_token_ids)
        seq_len = len(prompt) - 1
        self._greedy_forward_raw(
            {
                "input_ids": infinicore.from_list(
                    [prompt[1:]], dtype=infinicore.int64
                ),
                "position_ids": infinicore.from_list(
                    [list(range(seq_len))], dtype=infinicore.int64
                ),
                "past_kv_lengths": infinicore.from_list([0], dtype=infinicore.int32),
                "total_kv_lengths": infinicore.from_list(
                    [seq_len], dtype=infinicore.int32
                ),
                "input_offsets": infinicore.from_list(
                    [0, seq_len], dtype=infinicore.int32
                ),
                "cu_seqlens": infinicore.from_list([0, seq_len], dtype=infinicore.int32),
                "target_hidden_states": hidden_states.narrow(1, 0, seq_len),
                "sample_all_positions": False,
            }
        )

    def _draft_mtp_tokens(
        self,
        target_token,
        backbone_hidden,
        last_input_idx,
        base_len,
        chain_budget,
    ):
        """Chain the MTP head to draft tokens after the backbone's proposal.

        MTP at position `p` predicts token `p+2`. The first step anchors on the
        backbone hidden at `base_len - 1` and the just-proposed token; each
        later step chains from the previous MTP hidden, so the drafts form an
        autoregressive look-ahead.
        """
        drafts = []
        if chain_budget <= 0:
            return drafts
        current_hidden = backbone_hidden.narrow(1, last_input_idx, 1)  # h[base_len-1]
        for step in range(chain_budget):
            pos = base_len - 1 + step
            token = target_token if step == 0 else drafts[-1]
            out = self._greedy_forward_raw(
                {
                    "input_ids": infinicore.from_list(
                        [[token]], dtype=infinicore.int64
                    ),
                    "position_ids": infinicore.from_list(
                        [[pos]], dtype=infinicore.int64
                    ),
                    "past_kv_lengths": infinicore.from_list(
                        [pos], dtype=infinicore.int32
                    ),
                    "total_kv_lengths": infinicore.from_list(
                        [pos + 1], dtype=infinicore.int32
                    ),
                    "input_offsets": infinicore.from_list(
                        [0, 1], dtype=infinicore.int32
                    ),
                    "cu_seqlens": infinicore.from_list(
                        [0, pos + 1], dtype=infinicore.int32
                    ),
                    "target_hidden_states": current_hidden,
                }
            )
            drafts.append(int(out["output_ids"].to_numpy().tolist()[0]))
            current_hidden = out["hidden_states"]
        return drafts

    def _static_verify(self, combined, base_len):
        """Run the backbone over the speculative window and return its samples.

        The backbone cache already holds `[0, base_len-1]`; this writes
        `[base_len, base_len+len(combined)-1]`. Rejected positions' KV is never
        read again (every later forward starts at the accepted frontier and its
        causal mask extends only to the current position), so no rollback is
        needed for the static cache.
        """
        d = len(combined)
        out = self._greedy_forward_raw(
            {
                "input_ids": infinicore.from_list(
                    [combined], dtype=infinicore.int64
                ),
                "position_ids": infinicore.from_list(
                    [list(range(base_len, base_len + d))], dtype=infinicore.int64
                ),
                "past_kv_lengths": infinicore.from_list(
                    [base_len], dtype=infinicore.int32
                ),
                "total_kv_lengths": infinicore.from_list(
                    [base_len + d], dtype=infinicore.int32
                ),
                "input_offsets": infinicore.from_list(
                    [0, d], dtype=infinicore.int32
                ),
                "cu_seqlens": infinicore.from_list(
                    [0, base_len + d], dtype=infinicore.int32
                ),
            }
        )
        return out["output_ids"].to_numpy().tolist()

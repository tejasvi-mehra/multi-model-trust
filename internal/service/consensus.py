"""Claim grouping, opposition detection, and consensus labeling."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from internal.framework.logger import log_json
from internal.service.schemas import (
    ClaimValidation,
    ConsensusGroup,
    ConsensusMember,
    CrossLlmJudgePrompt,
    ModelResponse,
)


def _tokens(statement: str, stopwords: set[str]) -> set[str]:
    """Convert a claim into comparable tokens, skipping configured stopwords."""

    return {
        token
        for token in re.findall(r"[a-z0-9]+", statement.lower())
        if token not in stopwords
    }


def _similarity(left: str, right: str, stopwords: set[str]) -> float:
    """Measure Jaccard token overlap between two claim statements (0.0 to 1.0)."""

    left_tokens = _tokens(left, stopwords)
    right_tokens = _tokens(right, stopwords)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _has_negation(statement: str) -> bool:
    """Detect negation markers such as 'not' that often signal contradiction."""

    normalized = statement.lower()
    return bool(
        re.search(r"\b(no|not|never|cannot|can't|without|unsupported)\b", normalized)
    )


def _has_opposition(
    left: str,
    right: str,
    min_similarity: float,
    stopwords: set[str],
    opposing_terms: list[tuple[str, str]],
) -> bool:
    """Detect deterministic contradictions using overlap plus opposing term pairs."""

    left_lower = left.lower()
    right_lower = right.lower()
    if _similarity(left, right, stopwords) >= min_similarity and _has_negation(
        left
    ) != _has_negation(right):
        return True
    return any(
        (positive in left_lower and negative in right_lower)
        or (negative in left_lower and positive in right_lower)
        for positive, negative in opposing_terms
    )


@dataclass
class ConsensusEngine:
    """Groups validly cited claims and labels agreement, disagreement, or invalidity."""

    # All fields below are populated from env via config.build_consensus_engine().
    similarity_threshold: float = 0.35
    opposition_similarity_threshold: float = 0.35
    cross_llm_judge_enabled: bool = False
    cross_llm_judge_model_name: str = "LLM judge model"
    cross_llm_judge_borderline_low: float = 0.75
    cross_llm_judge_borderline_high: float = 0.85
    stopwords: set[str] = field(default_factory=set)
    opposing_terms: list[tuple[str, str]] = field(default_factory=list)

    def build(
        self,
        responses: list[ModelResponse],
        validations: list[ClaimValidation] | None = None,
        *,
        prompt: str = "",
        run_logger: logging.Logger | None = None,
    ) -> list[ConsensusGroup]:
        """Build consensus groups, prioritizing claims with valid citations."""

        logger = run_logger or logging.getLogger(__name__)
        enforce_citations = validations is not None
        validation_map = {
            (validation.model, validation.statement): validation
            for validation in validations or []
        }

        log_json(
            logger,
            "consensus.start",
            {
                "prompt": prompt,
                "model_count": len(responses),
                "enforce_citations": enforce_citations,
                "thresholds": self._threshold_payload(),
                "responses": [
                    {
                        "model": response.model,
                        "claims": [claim.statement for claim in response.claims],
                    }
                    for response in responses
                ],
                "citation_results": [
                    {
                        "model": validation.model,
                        "statement": validation.statement,
                        "citation_status": validation.citation_status,
                        "all_citations_valid": validation.all_citations_valid,
                        "invalid_reason": validation.invalid_reason,
                        "citations": [
                            {
                                "url": str(citation.url),
                                "quote": citation.quote,
                                "valid": citation.valid,
                                "quote_found": citation.quote_found,
                                "unreachable": citation.unreachable,
                                "support_score": citation.support_score,
                                "reason": citation.reason,
                            }
                            for citation in validation.citations
                        ],
                    }
                    for validation in validations or []
                ],
            },
        )

        valid_groups: list[list[ConsensusMember]] = []
        invalid_groups: list[list[ConsensusMember]] = []
        invalidated_by_valid_group: dict[int, list[ConsensusMember]] = {}
        judge_prompts: dict[int, CrossLlmJudgePrompt] = {}

        for response in responses:
            for claim in response.claims:
                validation = validation_map.get((response.model, claim.statement))
                member = self._build_member(
                    validation, response, claim, enforce_citations
                )

                log_json(
                    logger,
                    "consensus.claim_member",
                    {
                        "model": member.model,
                        "statement": member.statement,
                        "citation_status": member.citation_status,
                        "citation_errors": member.citation_errors,
                        "included_in_similarity": member.citation_status == "VALID",
                    },
                )

                if member.citation_status == "VALID":
                    group_index, judge_prompt = self._find_group(
                        valid_groups, claim.statement, logger
                    )
                    if group_index is None:
                        valid_groups.append([member])
                    else:
                        valid_groups[group_index].append(member)
                        if judge_prompt:
                            judge_prompts[group_index] = judge_prompt
                else:
                    group_index = self._find_valid_group_for_invalidated(
                        valid_groups, claim.statement, logger
                    )
                    if group_index is not None:
                        invalidated_by_valid_group.setdefault(group_index, []).append(
                            member
                        )
                        log_json(
                            logger,
                            "consensus.invalidated_for_valid_group",
                            {
                                "model": member.model,
                                "statement": member.statement,
                                "group_index": group_index,
                                "reason": member.citation_errors,
                            },
                        )
                    else:
                        invalid_group_index = self._find_group_index(
                            invalid_groups, claim.statement, logger
                        )
                        if invalid_group_index is None:
                            invalid_groups.append([member])
                        else:
                            invalid_groups[invalid_group_index].append(member)

        remaining_invalid_groups: list[list[ConsensusMember]] = []
        for invalid_group in invalid_groups:
            unassigned_invalid_members: list[ConsensusMember] = []
            for invalid_member in invalid_group:
                group_index = self._find_valid_group_for_invalidated(
                    valid_groups, invalid_member.statement, logger
                )
                if group_index is not None:
                    invalidated_by_valid_group.setdefault(group_index, []).append(
                        invalid_member
                    )
                else:
                    unassigned_invalid_members.append(invalid_member)
            if unassigned_invalid_members:
                remaining_invalid_groups.append(unassigned_invalid_members)

        consensus = [
            self._label_group(
                group,
                invalidated_by_valid_group.get(index, []),
                judge_prompts.get(index),
                logger,
                index,
            )
            for index, group in enumerate(valid_groups)
        ]
        consensus.extend(
            self._label_group(
                [],
                invalid_group,
                None,
                logger,
                index,
            )
            for index, invalid_group in enumerate(
                remaining_invalid_groups,
                start=len(valid_groups),
            )
        )

        log_json(
            logger,
            "consensus.complete",
            {
                "group_count": len(consensus),
                "labels": [group.label for group in consensus],
                "groups": [
                    {
                        "label": group.label,
                        "summary": group.summary,
                        "member_models": [member.model for member in group.members],
                        "invalidated_models": [
                            member.model for member in group.invalidated_members
                        ],
                        "cross_llm_judge_used": group.cross_llm_judge is not None,
                    }
                    for group in consensus
                ],
            },
        )
        return consensus

    def _build_member(
        self,
        validation: ClaimValidation | None,
        response: ModelResponse,
        claim,
        enforce_citations: bool,
    ) -> ConsensusMember:
        """Convert one claim plus validation into a consensus member."""

        return ConsensusMember(
            model=response.model,
            statement=claim.statement,
            citation_urls=[citation.url for citation in claim.citations],
            citation_status=(
                validation.citation_status
                if validation
                else ("MISSING" if enforce_citations else "VALID")
            ),
            citation_errors=self._citation_errors(validation, enforce_citations),
        )

    def _find_group(
        self,
        groups: list[list[ConsensusMember]],
        statement: str,
        logger: logging.Logger,
    ) -> tuple[int | None, CrossLlmJudgePrompt | None]:
        """Find a related valid-citation group and attach judge when needed."""

        for index, group in enumerate(groups):
            for member in group:
                comparison = self._compare_for_grouping(statement, member.statement)
                log_json(
                    logger,
                    "consensus.comparison",
                    {
                        "candidate_statement": statement,
                        "existing_statement": member.statement,
                        **comparison,
                    },
                )
                if comparison["grouped"]:
                    judge_data = comparison["judge_prompt"]
                    judge_prompt = (
                        CrossLlmJudgePrompt(**judge_data) if judge_data else None
                    )
                    return index, judge_prompt
        return None, None

    def _find_group_index(
        self,
        groups: list[list[ConsensusMember]],
        statement: str,
        logger: logging.Logger,
    ) -> int | None:
        """Find a related group for invalid or unreachable claims without judge."""

        for index, group in enumerate(groups):
            for member in group:
                comparison = self._compare_for_grouping(
                    statement,
                    member.statement,
                    allow_judge=False,
                )
                log_json(
                    logger,
                    "consensus.invalid_group_comparison",
                    {
                        "candidate_statement": statement,
                        "existing_statement": member.statement,
                        **comparison,
                    },
                )
                if comparison["grouped"]:
                    return index
        return None

    def _find_valid_group_for_invalidated(
        self,
        groups: list[list[ConsensusMember]],
        statement: str,
        logger: logging.Logger,
    ) -> int | None:
        """Link an invalid claim to a valid group using a looser relevancy threshold."""

        for index, group in enumerate(groups):
            for member in group:
                comparison = self._compare_for_grouping(
                    statement,
                    member.statement,
                    linking_threshold=self.opposition_similarity_threshold,
                )
                log_json(
                    logger,
                    "consensus.invalidated_link_comparison",
                    {
                        "invalid_statement": statement,
                        "valid_statement": member.statement,
                        **comparison,
                    },
                )
                if comparison["grouped"]:
                    return index
        return None

    def _compare_for_grouping(
        self,
        left: str,
        right: str,
        linking_threshold: float | None = None,
        *,
        allow_judge: bool = True,
    ) -> dict[str, Any]:
        """Score one pair of claims and decide whether they belong in one group."""

        similarity = _similarity(left, right, self.stopwords)
        opposition = _has_opposition(
            left,
            right,
            self.opposition_similarity_threshold,
            self.stopwords,
            self.opposing_terms,
        )
        active_threshold = (
            linking_threshold
            if linking_threshold is not None
            else self.similarity_threshold
        )
        threshold_match = similarity >= active_threshold
        consensus_reached = threshold_match or opposition
        judge_prompt = None
        borderline = False
        if allow_judge and not consensus_reached:
            judge_prompt = self._judge_if_borderline(left, right, similarity)
            borderline = judge_prompt is not None

        grouped = consensus_reached
        if (
            not grouped
            and judge_prompt is not None
            and judge_prompt.decision in {"AGREE", "DISAGREE"}
        ):
            grouped = True

        return {
            "similarity": round(similarity, 4),
            "similarity_threshold": active_threshold,
            "opposition_similarity_threshold": self.opposition_similarity_threshold,
            "threshold_match": threshold_match,
            "opposition_detected": opposition,
            "consensus_reached_without_judge": consensus_reached,
            "cross_llm_judge_enabled": self.cross_llm_judge_enabled,
            "borderline_low": self.cross_llm_judge_borderline_low,
            "borderline_high": self.cross_llm_judge_borderline_high,
            "borderline_match": borderline,
            "grouped": grouped,
            "judge_prompt": self._judge_payload(judge_prompt),
            "judge_decision": judge_prompt.decision if judge_prompt else None,
        }

    def _judge_payload(
        self,
        judge_prompt: CrossLlmJudgePrompt | None,
    ) -> dict[str, str] | None:
        """Serialize judge prompt data for JSON logs."""

        if judge_prompt is None:
            return None
        return {
            "model_name": judge_prompt.model_name,
            "prompt": judge_prompt.prompt,
            "decision": judge_prompt.decision,
        }

    def _label_group(
        self,
        members: list[ConsensusMember],
        invalidated_members: list[ConsensusMember],
        judge_prompt: CrossLlmJudgePrompt | None,
        logger: logging.Logger,
        group_index: int,
    ) -> ConsensusGroup:
        """Assign the final group label after citation filtering and comparison."""

        has_disagreement = any(
            _has_opposition(
                left.statement,
                right.statement,
                self.opposition_similarity_threshold,
                self.stopwords,
                self.opposing_terms,
            )
            or (judge_prompt is not None and judge_prompt.decision == "DISAGREE")
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        )

        if not members:
            label = "INVALID"
            summary = (
                "All related claims have missing, invalid, or unreachable citations."
            )
        elif has_disagreement:
            label = "DISAGREE"
            summary = "Models make materially opposing claims."
        elif len({member.model for member in members}) > 1:
            label = "AGREE"
            summary = "Multiple models make materially similar claims."
        elif invalidated_members:
            label = "PARTIAL"
            summary = "One validated model claim was prioritized over invalid sources."
        else:
            label = "PARTIAL"
            summary = "Only one model made this claim or overlap was limited."

        log_json(
            logger,
            "consensus.group_decision",
            {
                "group_index": group_index,
                "label": label,
                "summary": summary,
                "member_models": [member.model for member in members],
                "invalidated_models": [member.model for member in invalidated_members],
                "cross_llm_judge_used": judge_prompt is not None,
                "cross_llm_judge_decision": (
                    judge_prompt.decision if judge_prompt else None
                ),
            },
        )

        return ConsensusGroup(
            label=label,
            summary=summary,
            members=members,
            invalidated_members=invalidated_members,
            cross_llm_judge=judge_prompt,
        )

    def _judge_if_borderline(
        self,
        left: str,
        right: str,
        similarity: float,
    ) -> CrossLlmJudgePrompt | None:
        """Build the simulated judge prompt when similarity is in the borderline band."""

        if not self.cross_llm_judge_enabled:
            return None
        if not (
            self.cross_llm_judge_borderline_low
            <= similarity
            <= self.cross_llm_judge_borderline_high
        ):
            return None

        prompt = (
            "Given Statement 1 and Statement 2, do they completely agree, "
            "directly contradict, or simply present different context? Respond in "
            "exactly one word: AGREE, DISAGREE, or NEUTRAL.\n\n"
            f"Statement 1: {left}\n"
            f"Statement 2: {right}"
        )
        if _has_opposition(
            left,
            right,
            self.opposition_similarity_threshold,
            self.stopwords,
            self.opposing_terms,
        ):
            decision = "DISAGREE"
        elif similarity >= self.cross_llm_judge_borderline_low:
            decision = "AGREE"
        else:
            decision = "NEUTRAL"
        return CrossLlmJudgePrompt(
            model_name=self.cross_llm_judge_model_name,
            prompt=prompt,
            decision=decision,
        )

    def _citation_errors(
        self,
        validation: ClaimValidation | None,
        enforce_citations: bool,
    ) -> list[str]:
        """Collect human-readable citation errors for one claim."""

        if validation is None:
            return ["missing citation validation"] if enforce_citations else []
        if validation.citation_status == "MISSING":
            return [validation.invalid_reason or "missing citation"]
        return [
            citation.reason
            for citation in validation.citations
            if not citation.valid and citation.reason
        ]

    def _threshold_payload(self) -> dict[str, Any]:
        """Return active thresholds for structured logging."""

        return {
            "similarity_threshold": self.similarity_threshold,
            "opposition_similarity_threshold": self.opposition_similarity_threshold,
            "cross_llm_judge_enabled": self.cross_llm_judge_enabled,
            "cross_llm_judge_model_name": self.cross_llm_judge_model_name,
            "cross_llm_judge_borderline_low": self.cross_llm_judge_borderline_low,
            "cross_llm_judge_borderline_high": self.cross_llm_judge_borderline_high,
            "stopword_count": len(self.stopwords),
            "opposing_term_pair_count": len(self.opposing_terms),
        }

"""
Answer Synthesis & Response Generation Layer.

Public API
----------
- ``AnswerSynthesisChain`` — 6-step synthesis chain
- ``SynthesisResult``      — final output schema
- ``EvidenceAttribution``  — per-chunk citation
- ``ConfidenceLevel``      — High / Medium / Low enum
- ``synthesize()``         — convenience function for single-shot synthesis
"""

from synthesizer.synthesis_chain import AnswerSynthesisChain
from synthesizer.synthesis_schema import (
    ConfidenceLevel,
    ConsolidatedEvidence,
    EvidenceAttribution,
    SynthesisResult,
)

from retriever.pipeline import RetrievalPipeline
from retriever.schema import RetrievalResult


def synthesize(query: str, **kwargs) -> SynthesisResult:
    """Convenience function: run retrieval + synthesis on a single query.

    Parameters
    ----------
    query:
        The user's natural-language question.
    **kwargs:
        Passed to ``RetrievalPipeline.__init__()``.

    Returns
    -------
    SynthesisResult
    """
    pipeline = RetrievalPipeline(**kwargs)
    chain = AnswerSynthesisChain()
    try:
        retrieval_result = pipeline.retrieve(query)
        return chain.synthesize(query, retrieval_result)
    finally:
        pipeline.close()


__all__ = [
    "AnswerSynthesisChain",
    "SynthesisResult",
    "EvidenceAttribution",
    "ConfidenceLevel",
    "ConsolidatedEvidence",
    "synthesize",
]

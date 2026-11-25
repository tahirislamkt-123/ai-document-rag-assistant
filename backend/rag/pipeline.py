import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .answerability import AnswerabilityAssessment


ABSTENTION_TEXT = "I cannot find the answer in the provided documents."
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    citation: int
    document_id: str
    page: int
    chunk_id: str


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    abstained: bool
    sources: Tuple[Source, ...]
    citations: Tuple[int, ...]
    abstention_reason: Optional[str] = None
    retrieval_assessment: Optional[AnswerabilityAssessment] = None

    def to_dict(self):
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "sources": [
                {
                    "citation": source.citation,
                    "document_id": source.document_id,
                    "page": source.page,
                    "chunk_id": source.chunk_id,
                }
                for source in self.sources
            ],
            "citations": list(self.citations),
            "abstention_reason": self.abstention_reason,
            "retrieval_assessment": (
                None
                if self.retrieval_assessment is None
                else {
                    "sufficient": self.retrieval_assessment.sufficient,
                    "empirical_precision": (
                        self.retrieval_assessment.empirical_precision
                    ),
                    "calibration_id": (
                        self.retrieval_assessment.calibration_id
                    ),
                    "reason": self.retrieval_assessment.reason,
                }
            ),
        }


class RAGPipeline:
    def __init__(
        self,
        search,
        generator,
        top_k=3,
        max_prompt_tokens=1536,
        answerability_gate=None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")

        self.search = search
        self.generator = generator
        self.top_k = top_k
        self.max_prompt_tokens = max_prompt_tokens
        self.answerability_gate = answerability_gate

    def ask(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        hits = list(self.search(question, self.top_k) or [])[:self.top_k]
        if not hits:
            return self._abstain("no_retrieval_hits")

        candidate_sources = tuple(
            self._source_from_hit(hit, citation)
            for citation, hit in enumerate(hits, start=1)
        )

        if self.answerability_gate is None:
            return self._abstain(
                "answerability_not_calibrated",
                sources=candidate_sources,
            )

        assessment = self.answerability_gate.assess(question, hits)
        if not assessment.sufficient:
            return self._abstain(
                assessment.reason or "insufficient_retrieval_evidence",
                sources=candidate_sources,
                retrieval_assessment=assessment,
            )

        evidence, sources = self._fit_evidence(question, hits)
        if not evidence:
            return self._abstain(
                "evidence_exceeds_prompt_budget",
                sources=candidate_sources,
                retrieval_assessment=assessment,
            )

        generated = self.generator.answer(question, evidence).strip()
        if not generated:
            return self._abstain(
                "empty_generation",
                sources=sources,
                retrieval_assessment=assessment,
            )

        citations = tuple(
            sorted({int(value) for value in _CITATION_PATTERN.findall(generated)})
        )
        valid_citations = {source.citation for source in sources}
        if any(citation not in valid_citations for citation in citations):
            return self._abstain(
                "unsupported_citation",
                sources=sources,
                retrieval_assessment=assessment,
            )

        if ABSTENTION_TEXT.casefold() in generated.casefold():
            return self._abstain(
                "generator_abstained",
                sources=sources,
                retrieval_assessment=assessment,
            )
        if not citations:
            return self._abstain(
                "answer_missing_citation",
                sources=sources,
                retrieval_assessment=assessment,
            )

        return RAGResponse(
            answer=generated,
            abstained=False,
            sources=sources,
            citations=citations,
            retrieval_assessment=assessment,
        )

    def _fit_evidence(self, question, hits):
        empty_evidence = self._serialize_evidence([])
        if self._count_prompt_tokens(question, empty_evidence) > self.max_prompt_tokens:
            return "", ()

        records = []
        sources = []
        for citation, hit in enumerate(hits, start=1):
            text = hit.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each retrieval hit must contain non-empty text")

            source = self._source_from_hit(hit, citation)
            record = {"citation": citation, "text": text}
            proposed = records + [record]
            if self._fits(question, proposed):
                records.append(record)
                sources.append(source)
                continue

            prefix = self._truncate_to_fit(
                question=question,
                records=records,
                citation=citation,
                text=text,
            )
            if prefix.strip():
                records.append({"citation": citation, "text": prefix})
                sources.append(source)
            break

        if not records:
            return "", ()
        return self._serialize_evidence(records), tuple(sources)

    def _truncate_to_fit(self, question, records, citation, text):
        empty_record = records + [{"citation": citation, "text": ""}]
        remaining_tokens = self.max_prompt_tokens - self._count_prompt_tokens(
            question,
            self._serialize_evidence(empty_record),
        )
        if remaining_tokens <= 0:
            return ""

        text_tokens = self.generator.count_text_tokens(text)
        if not isinstance(text_tokens, int) or text_tokens < 0:
            raise ValueError("count_text_tokens must return a non-negative int")
        token_limit = min(text_tokens, remaining_tokens)
        while token_limit > 0:
            prefix = self.generator.truncate_text(text, token_limit)
            if not isinstance(prefix, str):
                raise ValueError("truncate_text must return a string")
            if not prefix.strip():
                return ""

            proposed = records + [
                {"citation": citation, "text": prefix}
            ]
            prompt_tokens = self._count_prompt_tokens(
                question,
                self._serialize_evidence(proposed),
            )
            if prompt_tokens <= self.max_prompt_tokens:
                return prefix
            token_limit -= max(
                1,
                prompt_tokens - self.max_prompt_tokens,
            )
        return ""

    def _fits(self, question, records):
        return (
            self._count_prompt_tokens(
                question,
                self._serialize_evidence(records),
            )
            <= self.max_prompt_tokens
        )

    def _count_prompt_tokens(self, question, evidence):
        count = self.generator.count_prompt_tokens(question, evidence)
        if not isinstance(count, int) or count < 0:
            raise ValueError("count_prompt_tokens must return a non-negative int")
        return count

    @staticmethod
    def _serialize_evidence(records):
        return json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _source_from_hit(hit, citation):
        document_id = hit.get("document_id") or hit.get("source")
        chunk_id = hit.get("chunk_id") or hit.get("id")
        page = hit.get("page")
        if document_id is None or chunk_id is None or page is None:
            raise ValueError(
                "Each retrieval hit must include document_id/source, page, "
                "and chunk_id/id metadata"
            )
        return Source(
            citation=citation,
            document_id=str(document_id),
            page=int(page),
            chunk_id=str(chunk_id),
        )

    @staticmethod
    def _abstain(reason, sources=(), retrieval_assessment=None):
        return RAGResponse(
            answer=ABSTENTION_TEXT,
            abstained=True,
            sources=tuple(sources),
            citations=(),
            abstention_reason=reason,
            retrieval_assessment=retrieval_assessment,
        )

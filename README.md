# LLM Council – Decision Engine

A minimal, auditable AI decision system inspired by Aonxi’s philosophy:
**AI should decide only when proof exists.**

This project implements a lightweight LLM Council:
**3 agents → 2 judges → 1 verifiable decision object**

---

## Why This Exists

Most AI systems summarize opinions.
This system produces **defensible decisions** with:
- Independent reasoning
- Structured comparison
- Safety gating
- Persistent audit logs

---

## Architecture

### Agents (Generate)
Three independent agents generate answers:
- Precise Agent (fact-first)
- Creative Agent (strategic)
- Conservative Agent (risk-aware)

Each agent outputs:
- Answer
- Reasoning
- Citations
- Timestamp

Agents never see each other.

---

### Judges (Compare)
Two judges:
- Never generate answers
- Compare agent responses using a fixed rubric
- Output winner, rationale, and confidence delta

Judges enforce **decision-by-comparison**, not consensus hallucination.

---

### Decision Object
Final output is a structured JSON object:

```json
{
  "final_answer": "...",
  "confidence": 0.74,
  "risks": [...],
  "citations": [...],
  "safety_level": "safe",
  "audit_hash": "..."
}

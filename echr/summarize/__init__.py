"""
echr.summarize — Generate concise summaries of ECHR case texts.

Responsibilities
----------------
- Produce 200-word and 500-word summaries of judgment outcome cases (for RETRIEVAL examples)
- Produce summaries of comm-phase cases (subject matter → summary for relevance checking)
- Support multiple models: GPT-OSS (primary), GPT-4o (legacy Art 3 results)

Inputs
------
  data/processed/article{N}/outcome_cases.pkl   — Facts + The Law columns
  data/processed/article{N}/comm_cases.pkl      — Subject Matter column

Outputs
-------
  data/processed/article{N}/outcome_summaries.pkl   — {Filename, 200 Word Summary, 500 Word Summary}
  data/processed/article{N}/comm_summaries.pkl      — {Filename, 200 Word Summary}

Current implementation:
  old/Summarize_Cases/summarize_cases.py
  — comm_prompt_generation(): uses Subject Matter column
  — outcome_prompt_generation(): uses Facts + The Law columns
  Note: prompts hardcode "Article 3" — must be parametrised for Art 6/8.
Migration: add --article and --model args; move prompt functions here.
"""

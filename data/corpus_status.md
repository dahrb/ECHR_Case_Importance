# Corpus Status — Judgment Text Extraction (Art 6 & 8)

*Generated 2026-09-01, after SLURM job 10397289 (full re-scrape, COMPLETED 12:48 BST).*
Source: `s4_extract_text_v2_1.py --judgment-only`, item IDs from `article_itemids/article{N}_cases.json`.

## Summary

The judgment-text scrape is **as complete as HUDOC allows**. Remaining gaps are
**genuine HUDOC unavailability, not scraper errors**: for the missing item IDs the
HUDOC HTML-conversion endpoint (`/app/conversion/docx/html/body`) returns
**HTTP 204 No Content** (verified on a random sample: 24/25 returned 204, 1 transient
502). These item IDs exist in the HUDOC search index / metadata but have no
retrievable HTML body — likely PDF-only, non-English-only, or metadata-only entries.

**Usable cases = those with BOTH a fact and a law section** (needed for
`outcome_cases.pkl` and GOLD/KG/RETRIEVAL conditions):

| | Item IDs | fact & law | Coverage |
|---|---|---|---|
| **Art 6** | 30,118 | **24,559** | 82% |
| **Art 8** | 6,765 | **5,898** | 87% |

## Per-branch counts

### Article 6

| Branch | Item IDs | Fact | Law | fact & law | Coverage |
|---|---|---|---|---|---|
| ADMISSIBILITYCOM | 7,583 | 7,515 | 7,574 | 7,513 | 99% |
| ADMISSIBILITY | 7,853 | 7,816 | 7,795 | 7,777 | 99% |
| DECGRANDCHAMBER | 17 | 17 | 17 | 17 | 100% |
| COMMITTEE | 3,212 | 2,605 | 2,633 | 2,602 | 81% |
| CHAMBER | 11,251 | 6,464 | 6,456 | 6,455 | 57% |
| GRANDCHAMBER | 202 | 197 | 196 | 195 | 97% |
| **Total** | **30,118** | **24,614** | **24,671** | **24,559** | **82%** |

### Article 8

| Branch | Item IDs | Fact | Law | fact & law | Coverage |
|---|---|---|---|---|---|
| ADMISSIBILITYCOM | 1,368 | 1,346 | 1,365 | 1,345 | 98% |
| ADMISSIBILITY | 2,407 | 2,389 | 2,384 | 2,370 | 98% |
| DECGRANDCHAMBER | 11 | 11 | 11 | 11 | 100% |
| COMMITTEE | 519 | 422 | 444 | 422 | 81% |
| CHAMBER | 2,342 | 1,638 | 1,723 | 1,634 | 70% |
| GRANDCHAMBER | 118 | 117 | 117 | 116 | 98% |
| **Total** | **6,765** | **5,923** | **6,044** | **5,898** | **87%** |

## Notes / caveats

- **CHAMBER is the main gap** (Art 6: 57%, Art 8: 70%). Confirmed as HUDOC 204s, not
  scraper failure. If higher coverage is needed, the only recovery path is an
  alternative HUDOC endpoint (PDF/DOCX conversion) or the raw metadata full-text — not
  a re-run of the current scraper.
- **Transient 502s** exist at the margin (~1/25 in sampling). A retry pass over the
  missing IDs would recover a small number; the bulk (204s) would not change.
- **`*_missing.txt` logs are NOT per-article.** `s4_extract_text_v2_1.py` writes
  `{DOCTYPE}_{section}_missing.txt` without an article prefix, so the Art 8 run
  overwrote Art 6's logs. Current logs reflect **Art 8 only**. Namespacing these by
  article is a small fix worth making before relying on them.
- **Comm-phase corpus** (`corpora/communication_phase/`) is article-agnostic and was
  populated separately (from the original zip) — not touched by this job. Validated
  against `important_labels.csv` (7,363 IDs): subject_matter 7,029 (95%),
  questions 7,234 (98%); 0 empty files, 0 stale files. Missing entries = the
  subject-matter / questions header is absent in the source document.

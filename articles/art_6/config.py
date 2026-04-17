"""
Article 6 – Right to a Fair Trial
Configuration for the ECHR data pipeline.
"""

ARTICLE_NUM = 6
ARTICLE_NAME = "Article 6"
ARTICLE_DESCRIPTION = "the right to a fair trial"

# HUDOC kpthesaurus IDs for Art. 6 sub-keywords
# Core Art. 6 header keywords (use as the minimum filter set)
KEYWORDS = [
    '445',  # (Art. 6) Right to a fair trial
    '13',   # (Art. 6) Administrative proceedings
    '40',   # (Art. 6) Civil proceedings
    '76',   # (Art. 6) Criminal proceedings
    '627',  # (Art. 6) Expulsion
    '626',  # (Art. 6) Extradition
    '7',    # (Art. 6-1) Access to court
    '136',  # (Art. 6-1) Fair hearing
    '406',  # (Art. 6-1) Reasonable time
    '180',  # (Art. 6-1) Impartial tribunal
    '181',  # (Art. 6-1) Independent tribunal
    '385',  # (Art. 6-1) Public hearing
    '388',  # (Art. 6-1) Public judgment
    '325',  # (Art. 6-2) Presumption of innocence
    '440',  # (Art. 6-3) Rights of defence
    '122',  # (Art. 6-3-d) Examination of witnesses
]

KEYWORD_DESCRIPTIONS = {
    '445': '(Art. 6) Right to a fair trial',
    '13':  '(Art. 6) Administrative proceedings',
    '40':  '(Art. 6) Civil proceedings',
    '76':  '(Art. 6) Criminal proceedings',
    '627': '(Art. 6) Expulsion',
    '626': '(Art. 6) Extradition',
    '7':   '(Art. 6-1) Access to court',
    '136': '(Art. 6-1) Fair hearing',
    '406': '(Art. 6-1) Reasonable time',
    '180': '(Art. 6-1) Impartial tribunal',
    '181': '(Art. 6-1) Independent tribunal',
    '385': '(Art. 6-1) Public hearing',
    '388': '(Art. 6-1) Public judgment',
    '325': '(Art. 6-2) Presumption of innocence',
    '440': '(Art. 6-3) Rights of defence',
    '122': '(Art. 6-3-d) Examination of witnesses',
}

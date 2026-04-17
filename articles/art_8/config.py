"""
Article 8 – Right to Respect for Private and Family Life
Configuration for the ECHR data pipeline.
"""

ARTICLE_NUM = 8
ARTICLE_NAME = "Article 8"
ARTICLE_DESCRIPTION = "the right to respect for private and family life"

# HUDOC kpthesaurus IDs for Art. 8 sub-keywords
KEYWORDS = [
    '451',  # (Art. 8) Right to respect for private and family life
    '628',  # (Art. 8) Expulsion
    '629',  # (Art. 8) Extradition
    '634',  # (Art. 8) Positive obligations
    '424',  # (Art. 8-1) Respect for correspondence
    '425',  # (Art. 8-1) Respect for family life
    '426',  # (Art. 8-1) Respect for home
    '429',  # (Art. 8-1) Respect for private life
    '203',  # (Art. 8-2) Interference
    '319',  # (Art. 8-2) In accordance with the law
    '268',  # (Art. 8-2) Necessary in a democratic society
]

KEYWORD_DESCRIPTIONS = {
    '451': '(Art. 8) Right to respect for private and family life',
    '628': '(Art. 8) Expulsion',
    '629': '(Art. 8) Extradition',
    '634': '(Art. 8) Positive obligations',
    '424': '(Art. 8-1) Respect for correspondence',
    '425': '(Art. 8-1) Respect for family life',
    '426': '(Art. 8-1) Respect for home',
    '429': '(Art. 8-1) Respect for private life',
    '203': '(Art. 8-2) Interference',
    '319': '(Art. 8-2) In accordance with the law',
    '268': '(Art. 8-2) Necessary in a democratic society',
}

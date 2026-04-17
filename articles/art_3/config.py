"""
Article 3 – Prohibition of Torture
Configuration for the ECHR data pipeline.
"""

ARTICLE_NUM = 3
ARTICLE_NAME = "Article 3"
ARTICLE_DESCRIPTION = "the prohibition of torture"

# HUDOC kpthesaurus IDs for all Art. 3 sub-keywords
KEYWORDS = ['350', '89', '90', '596', '620', '618', '192', '193', '633', '492']

KEYWORD_DESCRIPTIONS = {
    '350': '(Art. 3) Prohibition of torture',
    '89':  '(Art. 3) Degrading punishment',
    '90':  '(Art. 3) Degrading treatment',
    '596': '(Art. 3) Effective investigation',
    '620': '(Art. 3) Expulsion',
    '618': '(Art. 3) Extradition',
    '192': '(Art. 3) Inhuman punishment',
    '193': '(Art. 3) Inhuman treatment',
    '633': '(Art. 3) Positive obligations',
    '492': '(Art. 3) Torture',
}

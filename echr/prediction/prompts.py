"""
Article-aware prompt generation for ECHR importance prediction.
Ported and generalised from old/GPT_Experiments.py.
"""

ARTICLE_DESCRIPTIONS = {
    '3': "Article 3 of the European Convention of Human Rights, concerning the prohibition of torture",
    '6': "Article 6 of the European Convention of Human Rights, concerning the right to a fair trial",
    '8': "Article 8 of the European Convention of Human Rights, concerning the right to respect for private and family life",
}

IMPORTANCE_LEVELS = (
    "key_case: These are the most important and have been selected as key cases and have been "
    "selected for publication in the Court's official reports; "
    "1: The case is of high importance. The case makes a significant contribution to the development, "
    "clarification or modification of its case law, either generally or in relation to a particular case; "
    "2: The case is of medium importance. The case while not making a significant contribution to the "
    "case-law, nevertheless it goes beyond merely applying existing case law; "
    "3: The case is of low importance. The case is of limited interest and simply applies existing case law"
)

JSON_SCHEMA = '{"Case Importance": "string (select one of: key_case, 1, 2, 3)", "Reasoning": "string"}'

SUMMARIZE_SCHEMA = '{"200 Word Summary": "string", "500 Word Summary": "string"}'


def base_zero_shot_prompt(row, article: str, text: int = 1, cot: bool = False) -> str:
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    match text:
        case 1:
            text_content = row['Subject Matter']
            text_label = 'subject matter of the case'
        case 2:
            text_content = row['Questions']
            text_label = 'questions asked to the parties'
        case 3:
            text_content = str(row['Subject Matter']) + ' ' + str(row['Questions'])
            text_label = 'subject matter of the case and the questions asked to the parties'
        case _:
            raise ValueError(f"Invalid text value: {text}")

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to predict the "
        f"importance of a case, based on information provided from a communicated case. Importance "
        f"in a legal setting refers to the significance of a case in terms of its impact on the "
        f"development of case law.\n"
        f"All the cases concern {article_desc}.\n"
        f"You will be given a communicated case, including the {text_label}.\n"
        f"You are given a description of the different levels of importance: {IMPORTANCE_LEVELS}.\n"
        f"Based on the information given to you predict the importance of the case according to the "
        f"criteria given. If you do not know the importance, state that you do not have enough "
        f"information.{COT_SUFFIX if cot else ''}\n"
        f"The output should be given directly in JSON format, with the following schema: {JSON_SCHEMA}.\n"
        f"The communicated case information you should base your judgement on is as follows: {text_content}."
    )


COT_SUFFIX = (
    " Ensure when giving your reason you think through it step by step and provide a clear "
    "and concise explanation for your choice."
)

IMPORTANCE_LABEL_TO_KEY = {1: "key_case", 2: "1", 3: "2", 4: "3"}


def retrieval_prompt(row, article: str, examples: dict, text: int = 1) -> str:
    """RAG prompt. examples = {file_id: (importance_int, summary_str), ...}"""
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    match text:
        case 1:
            text_content = row['Subject Matter']
            text_label = 'subject matter of the case'
        case 2:
            text_content = row['Questions']
            text_label = 'questions asked to the parties'
        case 3:
            text_content = str(row['Subject Matter']) + ' ' + str(row['Questions'])
            text_label = 'subject matter of the case and the questions asked to the parties'
        case _:
            raise ValueError(f"Invalid text value: {text}")

    examples_str = ""
    for _fid, (imp, summary) in examples.items():
        imp_key = IMPORTANCE_LABEL_TO_KEY.get(int(imp), str(imp))
        examples_str += f"Summary: {summary}\nImportance level for that case: {imp_key}.\n\n"

    additional_context = (
        "You are also given summaries of a number of relevant outcome cases and their importance "
        f"levels; consider these cases carefully when making your decision.\n{examples_str}"
        if examples_str else ""
    )

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to predict the "
        f"importance of a case, based on information provided from a communicated case. Importance "
        f"in a legal setting refers to the significance of a case in terms of its impact on the "
        f"development of case law.\n"
        f"All the cases concern {article_desc}.\n"
        f"You will be given a communicated case, including the {text_label}.\n"
        f"You are given a description of the different levels of importance: {IMPORTANCE_LEVELS}.\n"
        f"{additional_context}"
        f"Based on the information given to you predict the importance of the case according to the "
        f"criteria given. If you do not know the importance, state that you do not have enough "
        f"information.\n"
        f"The output should be given directly in JSON format, with the following schema: {JSON_SCHEMA}.\n"
        f"The communicated case information you should base your judgement on is as follows: {text_content}."
    )


def few_shot_prompt(row, article: str, examples: dict, text: int = 1,
                    max_chars: int = None) -> str:
    """
    Few-shot prompt with fixed comm-phase examples (one per importance class).
    examples = {importance_int: subject_matter_str, ...} (keys 1..4)
    max_chars: if set, truncates each example and the test-case text to this many chars.
    """
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    match text:
        case 1:
            text_content = row['Subject Matter']
            text_label = 'subject matter of the case'
        case 2:
            text_content = row['Questions']
            text_label = 'questions asked to the parties'
        case 3:
            text_content = str(row['Subject Matter']) + ' ' + str(row['Questions'])
            text_label = 'subject matter of the case and the questions asked to the parties'
        case _:
            raise ValueError(f"Invalid text value: {text}")

    if max_chars is not None:
        text_content = str(text_content)[:max_chars]

    def _trunc(s):
        return str(s)[:max_chars] if max_chars is not None else str(s)

    label_names = {1: "key_case", 2: "Level 1", 3: "Level 2", 4: "Level 3"}
    ex_parts = "; ".join(
        f"{label_names[k]}: {_trunc(v)}"
        for k, v in sorted(examples.items())
        if v
    )
    additional_context = (
        f"You are also given a number of examples for each level of importance. {ex_parts}. "
        if ex_parts else ""
    )

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to predict the "
        f"importance of a case, based on information provided from a communicated case. Importance "
        f"in a legal setting refers to the significance of a case in terms of its impact on the "
        f"development of case law.\n"
        f"All the cases concern {article_desc}.\n"
        f"You will be given a communicated case, including the {text_label}.\n"
        f"You are given a description of the different levels of importance: {IMPORTANCE_LEVELS}.\n"
        f"{additional_context}"
        f"Based on the information given to you predict the importance of the case according to the "
        f"criteria given. If you do not know the importance, state that you do not have enough "
        f"information.\n"
        f"The output should be given directly in JSON format, with the following schema: {JSON_SCHEMA}.\n"
        f"The communicated case information you should base your judgement on is as follows: {text_content}."
    )


COURT_LEVELS = (
    "Judgment: In a judgment, the ECtHR will rule on the merits and/or just satisfaction, "
    "possibly in addition to the admissibility, of a complaint. The Court may also deliver a "
    "judgment striking out an application at a late stage in proceedings. "
    "Decision: In a decision, the ECtHR will confine its examination to the admissibility of an "
    "application only. If declared inadmissible, the Court issues a decision to that effect. If "
    "declared admissible, the Court proceeds to examine the merits and delivers a judgment."
)

COURT_SCHEMA = '{"Court": "string (Judgment or Decision)", "Reasoning": "string"}'

COURT_MAP = {"judgment": 1, "decision": 2}


def court_prompt(row, article: str, text: int = 1) -> str:
    """Predict whether the communicated case will result in a Judgment or Decision."""
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    match text:
        case 1:
            text_content = row['Subject Matter']
            text_label = 'subject matter of the case'
        case 2:
            text_content = row['Questions']
            text_label = 'questions asked to the parties'
        case 3:
            text_content = str(row['Subject Matter']) + ' ' + str(row['Questions'])
            text_label = 'subject matter of the case and the questions asked to the parties'
        case _:
            raise ValueError(f"Invalid text value: {text}")

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to predict "
        f"whether a communicated case will result in a Judgment or a Decision.\n"
        f"All the cases concern {article_desc}.\n"
        f"You will be given a communicated case, including the {text_label}.\n"
        f"You are given a description of the court outcome types: {COURT_LEVELS}.\n"
        f"Based only on the information given to you, predict whether the case will result in a "
        f"Judgment or a Decision. If you do not know, state that you do not have enough information.\n"
        f"The output should be given directly in JSON format, with the following schema: {COURT_SCHEMA}.\n"
        f"The communicated case information you should base your judgement on is as follows: {text_content}."
    )


def summarize_outcome_prompt(row, article: str) -> str:
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    try:
        law = row['The Law']
    except (KeyError, AttributeError):
        law = "The Law section is not available for this case."

    facts = row.get('Facts', 'The Facts section is not available for this case.')

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to summarise "
        f"outcome cases.\n"
        f"You will be given the 'Facts' and 'The Law' sections of an outcome case. Please provide "
        f"two summaries of the facts of the case, one with a maximum word count of 200 words and "
        f"another with a maximum word count of 500 words. The summaries should be concise and capture "
        f"the key aspects of the case.\n"
        f"The case relates to {article_desc}.\n"
        f"The 'Facts' section of the case is: {facts}.\n"
        f"The 'The Law' section of the case is: {law}.\n"
        f"The output should be given directly in JSON format, with the following schema: "
        f"{SUMMARIZE_SCHEMA}."
    )

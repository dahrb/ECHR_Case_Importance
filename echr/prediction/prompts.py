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


def base_zero_shot_prompt(row, article: str, text: int = 1, cot: bool = False,
                          max_chars: int = None) -> str:
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


def retrieval_prompt(row, article: str, examples: dict, text: int = 1,
                     max_chars: int = None) -> str:
    """RAG prompt. examples = {file_id: (importance_int, summary_str), ...}.

    ``max_chars`` limits both the query text and each retrieved summary so the
    caller can enforce an aggregate context budget for larger values of k.
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

    examples_str = ""
    for _fid, (imp, summary) in examples.items():
        imp_key = IMPORTANCE_LABEL_TO_KEY.get(int(imp), str(imp))
        summary_text = str(summary)[:max_chars] if max_chars is not None else str(summary)
        examples_str += f"Summary: {summary_text}\nImportance level for that case: {imp_key}.\n\n"

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
    "Committee: A Committee consists of 3 judges. The Committee can rule on the merits of a case "
    "where the Court's case law is well established. They can also rule on the admissibility of a "
    "case with well established case law. "
    "Chamber: A Chamber consists of 7 judges. The Chamber can decide on the merits or admissibility "
    "of a case if no prior decision has been reached by the Committee or a single judge. The Chamber "
    "can relinquish jurisdiction to the Grand Chamber. "
    "Grand Chamber: The Grand Chamber consists of 17 judges. It examines the cases that are submitted "
    "to it either after a Chamber has relinquished jurisdiction or when a request for referral of the "
    "case has been granted by the Grand Chamber Panel. The Grand Chamber deals with cases which raise "
    "a serious question affecting the interpretation of the Convention or if there is a risk that its "
    "resolution of the case would be inconsistent with a judgment previously delivered by the Court."
)

COURT_SCHEMA = '{"Court": "string (Committee, Chamber or Grand Chamber)", "Reasoning": "string"}'

# Maps lower-cased model output → integer label (1=Committee, 2=Chamber, 3=Grand Chamber)
COURT_MAP = {"committee": 1, "chamber": 2, "grand chamber": 3}

# Ground truth: source_file → integer label (mirrors old/PREDICTION/court_labels_data.ipynb mapping)
# ADMISSIBILITYCOM/COMMITTEE → 1, CHAMBER/ADMISSIBILITY → 2, GRANDCHAMBER/DECGRANDCHAMBER → 3
COURT_SOURCE_MAP = {
    "pruned_COMMITTEE_meta.json": 1,
    "pruned_ADMISSIBILITYCOM_meta.json": 1,
    "pruned_CHAMBER_meta.json": 2,
    "pruned_ADMISSIBILITY_meta.json": 2,
    "pruned_GRANDCHAMBER_meta.json": 3,
    "pruned_DECGRANDCHAMBER_meta.json": 3,
}


def court_prompt(row, article: str, text: int = 1, max_chars: int = None) -> str:
    """Predict the court formation level: Committee, Chamber, or Grand Chamber."""
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

    return (
        f"You are a lawyer in the European Court of Human Rights, and your goal is to predict "
        f"whether a case will end up at the level of Committee, Chamber or Grand Chamber; "
        f"based on information provided from a communicated case.\n"
        f"All the cases concern {article_desc}.\n"
        f"You will be given a communicated case, including the {text_label}.\n"
        f"You are given a description of the different levels of the court: {COURT_LEVELS}.\n"
        f"Based only on the information given to you, predict the court level. "
        f"If you do not know, state that you do not have enough information.\n"
        f"The output should be given directly in JSON format, with the following schema: {COURT_SCHEMA}.\n"
        f"The communicated case information you should base your judgement on is as follows: {text_content}."
    )


ITER_LEVELS = {
    "key_case": (
        "key_case (most important): selected as a key case and published in the Court's official "
        "reports; makes a landmark contribution to the development of case law"
    ),
    "1": (
        "1 (high importance): makes a significant contribution to the development, clarification "
        "or modification of case law, either generally or in relation to a particular issue"
    ),
    "2": (
        "2 (medium importance): goes beyond merely applying existing case law, but does not make "
        "a significant contribution to its development"
    ),
    "3": (
        "3 (low importance): of limited interest; simply applies existing case law"
    ),
}

ITER_SCHEMA = '{"Level": "string (the importance level being assessed)", "Matches": "string (Yes or No)", "Confidence": "number (0.0–1.0, how confident you are in the Matches answer)", "Reasoning": "string"}'


def iterative_prompt(row, article: str, level_key: str, text: int = 1,
                     max_chars: int = None) -> str:
    """
    Per-level binary query for iterative prompting (Exp 2).
    Ask the model whether the case matches one specific importance level.
    Run for each of {key_case, 1, 2, 3}; take argmax of Confidence over 'Yes' answers.
    """
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    level_desc = ITER_LEVELS[level_key]
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

    return (
        f"You are a lawyer in the European Court of Human Rights. Your task is to assess whether a "
        f"communicated case matches a specific importance level.\n"
        f"All cases concern {article_desc}.\n"
        f"You will be given the {text_label} of a communicated case.\n"
        f"The communicated case information: {text_content}.\n"
        f"The importance level to assess: {level_desc}.\n"
        f"Does this communicated case match this importance level? Answer Yes or No, and provide a "
        f"confidence score between 0.0 (not at all) and 1.0 (certain), along with brief reasoning.\n"
        f"The output must be in JSON format with this schema: {ITER_SCHEMA}.\n"
    )


def iterative_retrieval_prompt(row, article: str, level_key: str, examples: dict,
                               text: int = 1, max_chars: int = None) -> str:
    """Render one level-specific iterative prompt with the normal RAG context.

    The retrieved examples and their gold importance labels are rendered exactly
    as in :func:`retrieval_prompt`; only the decision instruction and response
    schema differ.  This permits the iterative arm to use the same
    article/retriever/k matrix as fine-tuned inference.
    """
    article_desc = ARTICLE_DESCRIPTIONS.get(str(article), f"Article {article} of the ECHR")
    level_desc = ITER_LEVELS[level_key]
    match text:
        case 1:
            text_content = row["Subject Matter"]
            text_label = "subject matter of the case"
        case 2:
            text_content = row["Questions"]
            text_label = "questions asked to the parties"
        case 3:
            text_content = str(row["Subject Matter"]) + " " + str(row["Questions"])
            text_label = "subject matter of the case and the questions asked to the parties"
        case _:
            raise ValueError(f"Invalid text value: {text}")

    if max_chars is not None:
        text_content = str(text_content)[:max_chars]

    examples_str = ""
    for _fid, (importance, summary) in examples.items():
        importance_key = IMPORTANCE_LABEL_TO_KEY.get(int(importance), str(importance))
        summary_text = str(summary)[:max_chars] if max_chars is not None else str(summary)
        examples_str += (
            f"Summary: {summary_text}\n"
            f"Importance level for that case: {importance_key}.\n\n"
        )

    additional_context = (
        "You are also given summaries of relevant outcome cases and their importance "
        f"levels; consider them carefully when making your decision.\n{examples_str}"
        if examples_str else ""
    )
    return (
        "You are a lawyer in the European Court of Human Rights. Your task is to assess "
        "whether a communicated case matches a specific importance level.\n"
        f"All cases concern {article_desc}.\n"
        f"You will be given the {text_label} of a communicated case.\n"
        f"{additional_context}"
        # Keep all expensive retrieved context and case text before the one
        # level-specific suffix.  The four iterative calls for a case can then
        # share vLLM's automatic prefix cache without changing their content.
        f"The communicated case information: {text_content}.\n"
        f"The importance level to assess: {level_desc}.\n"
        "Does this communicated case match this importance level? Answer Yes or No, and provide "
        "a confidence score between 0.0 (not at all) and 1.0 (certain), along with brief reasoning.\n"
        f"The output must be in JSON format with this schema: {ITER_SCHEMA}.\n"
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

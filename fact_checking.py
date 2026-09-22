import os
import re
import sys
from typing import Optional

import numpy as np
import requests
from sentence_transformers import CrossEncoder


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

# Ordine delle classi del modello NLI:
# 0 = contradiction
# 1 = entailment
# 2 = neutral
LABEL_MAP = {
    0: "REFUTES",
    1: "SUPPORTS",
    2: "NOT ENOUGH INFO",
}

ENTAILMENT_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.80
MAX_WIKIPEDIA_PAGES = 3
MIN_SENTENCE_LENGTH = 20


def softmax(values):
    values = np.array(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "FactCheckDemo/1.0 "
            "(educational project; contact@example.com)"
        )
    }
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_main_subject(claim: str) -> Optional[str]:
    claim = claim.strip()

    # Rimuove l'articolo iniziale "The" per cercare meglio su Wikipedia.
    claim_without_article = re.sub(
        r"^the\s+",
        "",
        claim,
        flags=re.IGNORECASE,
    )

    verbs = [
        r"\s+is\s+",
        r"\s+was\s+",
        r"\s+are\s+",
        r"\s+were\s+",
        r"\s+has\s+",
        r"\s+have\s+",
        r"\s+had\s+",
        r"\s+does\s+",
        r"\s+do\s+",
        r"\s+did\s+",
        r"\s+can\s+",
        r"\s+could\s+",
        r"\s+will\s+",
        r"\s+would\s+",
    ]

    pattern = "|".join(verbs)
    match = re.search(pattern, claim_without_article, flags=re.IGNORECASE)

    if match:
        subject = claim_without_article[:match.start()].strip()
    else:
        subject = " ".join(claim_without_article.split()[:4])

    subject = re.sub(r"[^A-Za-z0-9\s\-']", " ", subject)
    subject = " ".join(subject.split())

    return subject if subject else None


def sentence_mentions_subject(sentence: str, subject: str) -> bool:
    normalized_sentence = normalize_text(sentence)
    normalized_subject = normalize_text(subject)

    if not normalized_subject:
        return False

    if normalized_subject in normalized_sentence:
        return True

    subject_tokens = normalized_subject.split()

    if len(subject_tokens) == 1:
        return subject_tokens[0] in normalized_sentence.split()

    matched_tokens = sum(
        token in normalized_sentence.split()
        for token in subject_tokens
    )

    return matched_tokens >= max(1, len(subject_tokens) - 1)


def fetch_wikipedia_evidence(claim: str) -> list[tuple[str, str]]:
    subject = extract_main_subject(claim)

    if not subject:
        print("Impossibile estrarre il soggetto dalla claim.")
        return []

    print(f"Soggetto estratto: {subject}")

    wikipedia_api = "https://en.wikipedia.org/w/api.php"

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": subject,
        "format": "json",
        "srlimit": 5,
        "utf8": 1,
    }

    try:
        response = SESSION.get(
            wikipedia_api,
            params=search_params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        print(f"Errore durante la ricerca Wikipedia: {error}")
        return []

    results = data.get("query", {}).get("search", [])

    if not results:
        print(f"Nessun risultato Wikipedia per: {subject}")
        return []

    titles = [result["title"] for result in results[:5]]

    print("Titoli trovati:", ", ".join(titles))

    exact_match = None
    other_titles = []

    for title in titles:
        if normalize_text(title) == normalize_text(subject):
            exact_match = title
        else:
            other_titles.append(title)

    if exact_match:
        selected_titles = [exact_match] + other_titles[:MAX_WIKIPEDIA_PAGES - 1]
    else:
        selected_titles = titles[:MAX_WIKIPEDIA_PAGES]

    evidence = []

    for title in selected_titles:
        extract_params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exlimit": 1,
        }

        try:
            response = SESSION.get(
                wikipedia_api,
                params=extract_params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), None)

            if page is None:
                continue

            page_title = page.get("title", title)
            extract = page.get("extract", "").strip()

            if extract:
                evidence.append((page_title, extract))

        except requests.RequestException as error:
            print(f"Errore nel recupero della pagina '{title}': {error}")

    return evidence


def get_sentence_pairs(
    evidence: list[tuple[str, str]],
    subject: str,
) -> list[tuple[str, str]]:
    pairs = []

    for title, excerpt in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", excerpt)

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            if sentence_mentions_subject(sentence, subject):
                pairs.append((title, sentence))

    return pairs


def verify_claim(claim: str) -> None:
    print("\n🔍 Ricerca evidenza su Wikipedia...")

    subject = extract_main_subject(claim)

    if not subject:
        print("Impossibile identificare il soggetto.")
        print("VERDETTO: NOT ENOUGH INFO")
        return

    evidence = fetch_wikipedia_evidence(claim)

    if not evidence:
        print("Nessuna evidenza trovata su Wikipedia.")
        print("VERDETTO: NOT ENOUGH INFO")
        return

    print(f"Trovate {len(evidence)} pagine candidate.\n")

    sentence_pairs = get_sentence_pairs(evidence, subject)

    if not sentence_pairs:
        print("Nessuna frase pertinente al soggetto trovata.")
        print("VERDETTO: NOT ENOUGH INFO")
        return

    print(f"Frasi pertinenti da analizzare: {len(sentence_pairs)}")
    print("⏳ Caricamento modello NLI...")

    model = CrossEncoder(MODEL_NAME)

    best_entailment_score = -1.0
    best_contradiction_score = -1.0
    best_entailment_pair = None
    best_contradiction_pair = None

    for title, sentence in sentence_pairs:
        # L'ordine è importante:
        # prima premise/evidenza, poi hypothesis/claim.
        logits = model.predict([(sentence, claim)])[0]
        probabilities = softmax(logits)

        contradiction_score = float(probabilities[0])
        entailment_score = float(probabilities[1])
        neutral_score = float(probabilities[2])

        candidate = (
            title,
            sentence,
            contradiction_score,
            entailment_score,
            neutral_score,
        )

        if entailment_score > best_entailment_score:
            best_entailment_score = entailment_score
            best_entailment_pair = candidate

        if contradiction_score > best_contradiction_score:
            best_contradiction_score = contradiction_score
            best_contradiction_pair = candidate

    if best_entailment_pair is None or best_contradiction_pair is None:
        print("Non è stato possibile classificare le frasi trovate.")
        print("VERDETTO: NOT ENOUGH INFO")
        return

    if best_entailment_score >= ENTAILMENT_THRESHOLD:
        title, sentence, contradiction, entailment, neutral = best_entailment_pair
        verdict = "SUPPORTS"
        confidence = entailment

    elif best_contradiction_score >= CONTRADICTION_THRESHOLD:
        title, sentence, contradiction, entailment, neutral = best_contradiction_pair
        verdict = "REFUTES"
        confidence = contradiction

    else:
        title, sentence, contradiction, entailment, neutral = best_entailment_pair
        verdict = "NOT ENOUGH INFO"
        confidence = neutral

    print("\nEVIDENZA MIGLIORE:")
    print(f"Pagina: {title}")
    print(f"Frase: {sentence}\n")

    print("VERDETTO:", verdict)
    print("Confidenza:", round(confidence, 4))
    print("Contradiction:", round(contradiction, 4))
    print("Entailment:", round(entailment, 4))
    print("Neutral:", round(neutral, 4))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Uso: python .\\fact_checking.py "<claim>"')
        sys.exit(1)

    claim = sys.argv[1].strip()

    if not claim:
        print("La claim non può essere vuota.")
        sys.exit(1)

    verify_claim(claim)
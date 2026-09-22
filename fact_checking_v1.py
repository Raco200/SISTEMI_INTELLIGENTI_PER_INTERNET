import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sentence_transformers import CrossEncoder


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

# Ordine classi del modello:
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

DEFAULT_INPUT_FILE = "claims_100_labeled.csv"
DEFAULT_OUTPUT_FILE = "results.csv"


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "FactCheckingStudentProject/1.0 "
            "(educational use; contact@example.com)"
        )
    }
)


def softmax(values):
    values = np.array(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_main_subject(claim: str) -> Optional[str]:
    claim = claim.strip()

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

    sentence_tokens = set(normalized_sentence.split())
    subject_tokens = normalized_subject.split()

    if len(subject_tokens) == 1:
        return subject_tokens[0] in sentence_tokens

    matching_tokens = sum(
        token in sentence_tokens
        for token in subject_tokens
    )

    return matching_tokens >= len(subject_tokens) - 1


def fetch_wikipedia_evidence(
    claim: str,
    verbose: bool = True,
) -> list[tuple[str, str]]:
    subject = extract_main_subject(claim)

    if not subject:
        if verbose:
            print("Impossibile estrarre il soggetto dalla claim.")
        return []

    if verbose:
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
        if verbose:
            print(f"Errore durante la ricerca Wikipedia: {error}")
        return []

    results = data.get("query", {}).get("search", [])

    if not results:
        if verbose:
            print(f"Nessun risultato Wikipedia per: {subject}")
        return []

    titles = [result["title"] for result in results[:5]]

    if verbose:
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
            if verbose:
                print(
                    f"Errore nel recupero della pagina "
                    f"'{title}': {error}"
                )

    return evidence


def get_relevant_sentences(
    evidence: list[tuple[str, str]],
    subject: str,
) -> list[tuple[str, str]]:
    relevant_sentences = []

    for title, excerpt in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", excerpt)

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            if sentence_mentions_subject(sentence, subject):
                relevant_sentences.append((title, sentence))

    return relevant_sentences


def load_model() -> CrossEncoder:
    print("\nCaricamento del modello NLI...")
    print("Al primo avvio il download può richiedere qualche minuto.")

    return CrossEncoder(MODEL_NAME)


def verify_claim(
    claim: str,
    model: CrossEncoder,
    verbose: bool = True,
) -> dict:
    result = {
        "claim": claim,
        "predicted_label": "NOT ENOUGH INFO",
        "confidence": 0.0,
        "contradiction": 0.0,
        "entailment": 0.0,
        "neutral": 1.0,
        "page": "",
        "evidence": "",
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"CLAIM: {claim}")
        print("=" * 70)
        print("Ricerca evidenza su Wikipedia...")

    subject = extract_main_subject(claim)

    if not subject:
        if verbose:
            print("Impossibile identificare il soggetto.")
            print("VERDETTO: NOT ENOUGH INFO")
        return result

    evidence = fetch_wikipedia_evidence(claim, verbose=verbose)

    if not evidence:
        if verbose:
            print("Nessuna evidenza trovata su Wikipedia.")
            print("VERDETTO: NOT ENOUGH INFO")
        return result

    relevant_sentences = get_relevant_sentences(evidence, subject)

    if not relevant_sentences:
        if verbose:
            print("Nessuna frase pertinente al soggetto trovata.")
            print("VERDETTO: NOT ENOUGH INFO")
        return result

    if verbose:
        print(f"Pagine candidate: {len(evidence)}")
        print(f"Frasi pertinenti da analizzare: {len(relevant_sentences)}")

    best_entailment_score = -1.0
    best_contradiction_score = -1.0

    best_entailment_pair = None
    best_contradiction_pair = None

    for title, sentence in relevant_sentences:
        logits = model.predict([(sentence, claim)])[0]
        probabilities = softmax(logits)

        contradiction_score = float(probabilities[0])
        entailment_score = float(probabilities[1])
        neutral_score = float(probabilities[2])

        candidate = {
            "page": title,
            "evidence": sentence,
            "contradiction": contradiction_score,
            "entailment": entailment_score,
            "neutral": neutral_score,
        }

        if entailment_score > best_entailment_score:
            best_entailment_score = entailment_score
            best_entailment_pair = candidate

        if contradiction_score > best_contradiction_score:
            best_contradiction_score = contradiction_score
            best_contradiction_pair = candidate

    if best_entailment_pair is None or best_contradiction_pair is None:
        if verbose:
            print("Non è stato possibile classificare l'evidenza.")
            print("VERDETTO: NOT ENOUGH INFO")
        return result

    if best_entailment_score >= ENTAILMENT_THRESHOLD:
        result = {
            "claim": claim,
            "predicted_label": "SUPPORTS",
            "confidence": best_entailment_pair["entailment"],
            "contradiction": best_entailment_pair["contradiction"],
            "entailment": best_entailment_pair["entailment"],
            "neutral": best_entailment_pair["neutral"],
            "page": best_entailment_pair["page"],
            "evidence": best_entailment_pair["evidence"],
        }

    elif best_contradiction_score >= CONTRADICTION_THRESHOLD:
        result = {
            "claim": claim,
            "predicted_label": "REFUTES",
            "confidence": best_contradiction_pair["contradiction"],
            "contradiction": best_contradiction_pair["contradiction"],
            "entailment": best_contradiction_pair["entailment"],
            "neutral": best_contradiction_pair["neutral"],
            "page": best_contradiction_pair["page"],
            "evidence": best_contradiction_pair["evidence"],
        }

    else:
        result = {
            "claim": claim,
            "predicted_label": "NOT ENOUGH INFO",
            "confidence": best_entailment_pair["neutral"],
            "contradiction": best_entailment_pair["contradiction"],
            "entailment": best_entailment_pair["entailment"],
            "neutral": best_entailment_pair["neutral"],
            "page": best_entailment_pair["page"],
            "evidence": best_entailment_pair["evidence"],
        }

    if verbose:
        print("\nEVIDENZA MIGLIORE:")
        print(f"Pagina: {result['page']}")
        print(f"Frase: {result['evidence']}\n")

        print("VERDETTO:", result["predicted_label"])
        print("Confidenza:", round(result["confidence"], 4))
        print("Contradiction:", round(result["contradiction"], 4))
        print("Entailment:", round(result["entailment"], 4))
        print("Neutral:", round(result["neutral"], 4))

    return result


def read_claims_from_csv(file_path: str) -> list[dict]:
    claims = []

    with open(
        file_path,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or "claim" not in reader.fieldnames:
            raise ValueError(
                "Il CSV deve contenere almeno una colonna chiamata 'claim'."
            )

        for index, row in enumerate(reader, start=1):
            claim = row.get("claim", "").strip()

            if not claim:
                continue

            claims.append(
                {
                    "id": row.get("id", str(index)).strip(),
                    "claim": claim,
                    "gold_label": row.get("gold_label", "").strip().upper(),
                }
            )

    return claims


def write_results_to_csv(results: list[dict], output_file: str) -> None:
    fieldnames = [
        "id",
        "claim",
        "gold_label",
        "predicted_label",
        "is_correct",
        "confidence",
        "contradiction",
        "entailment",
        "neutral",
        "page",
        "evidence",
    ]

    with open(
        output_file,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)


def verify_claims_from_file(model: CrossEncoder) -> None:
    print("\n" + "=" * 70)
    print("VERIFICA DI CLAIM DA FILE CSV")
    print("=" * 70)

    file_name = input(
        f"Nome del file CSV "
        f"[Invio = {DEFAULT_INPUT_FILE}]: "
    ).strip()

    if not file_name:
        file_name = DEFAULT_INPUT_FILE

    file_path = Path(file_name)

    if not file_path.exists():
        print(f"\nErrore: il file '{file_name}' non esiste.")
        print("Metti il CSV nella stessa cartella di fact_checking.py.")
        return

    try:
        claims = read_claims_from_csv(str(file_path))
    except (OSError, ValueError) as error:
        print(f"\nErrore nella lettura del CSV: {error}")
        return

    if not claims:
        print("\nIl file non contiene claim valide.")
        return

    print(f"\nClaim lette dal file: {len(claims)}")
    print("L'elaborazione può richiedere tempo perché ogni claim")
    print("richiede ricerche Wikipedia e classificazione NLI.\n")

    results = []
    correct_predictions = 0
    labeled_claims = 0

    for index, item in enumerate(claims, start=1):
        print("\n" + "-" * 70)
        print(f"Elaborazione claim {index}/{len(claims)}")
        print("-" * 70)

        prediction = verify_claim(
            item["claim"],
            model,
            verbose=False,
        )

        gold_label = item["gold_label"]

        is_correct = ""

        if gold_label:
            labeled_claims += 1
            is_correct = prediction["predicted_label"] == gold_label

            if is_correct:
                correct_predictions += 1

        output_row = {
            "id": item["id"],
            "claim": item["claim"],
            "gold_label": gold_label,
            "predicted_label": prediction["predicted_label"],
            "is_correct": is_correct,
            "confidence": round(prediction["confidence"], 4),
            "contradiction": round(prediction["contradiction"], 4),
            "entailment": round(prediction["entailment"], 4),
            "neutral": round(prediction["neutral"], 4),
            "page": prediction["page"],
            "evidence": prediction["evidence"],
        }

        results.append(output_row)

        print(f"Claim: {item['claim']}")
        print(f"Predizione: {prediction['predicted_label']}")

        if gold_label:
            print(f"Etichetta corretta: {gold_label}")
            print(f"Corretta: {'SI' if is_correct else 'NO'}")

    write_results_to_csv(results, DEFAULT_OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("ELABORAZIONE COMPLETATA")
    print("=" * 70)
    print(f"Risultati salvati nel file: {DEFAULT_OUTPUT_FILE}")

    if labeled_claims > 0:
        accuracy = correct_predictions / labeled_claims

        print(f"Claim con etichetta: {labeled_claims}")
        print(f"Predizioni corrette: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")


def verify_single_claim(model: CrossEncoder) -> None:
    print("\n" + "=" * 70)
    print("VERIFICA DI UNA SINGOLA CLAIM")
    print("=" * 70)

    claim = input("\nScrivi la claim in inglese: ").strip()

    if not claim:
        print("La claim non può essere vuota.")
        return

    verify_claim(claim, model, verbose=True)


def print_menu() -> None:
    print("\n" + "=" * 70)
    print("SISTEMA DI FACT-CHECKING CON WIKIPEDIA E NLI")
    print("=" * 70)
    print("1 - Scrivere e verificare una singola claim")
    print("2 - Verificare claim da un file CSV")
    print("3 - Uscire")


def main() -> None:
    model = load_model()

    while True:
        print_menu()

        choice = input("\nScegli un'opzione (1, 2, 3): ").strip()

        if choice == "1":
            verify_single_claim(model)

        elif choice == "2":
            verify_claims_from_file(model)

        elif choice == "3":
            print("\nChiusura del programma.")
            sys.exit(0)

        else:
            print("\nScelta non valida. Inserisci 1, 2 oppure 3.")


if __name__ == "__main__":
    main()
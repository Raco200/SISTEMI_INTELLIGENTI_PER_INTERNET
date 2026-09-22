import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sentence_transformers import CrossEncoder


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

# Soglie iniziali. Puoi sperimentare con questi valori.
ENTAILMENT_THRESHOLD = 0.45
CONTRADICTION_THRESHOLD = 0.45

MAX_WIKIPEDIA_PAGES = 5
MAX_SENTENCES_PER_CLAIM = 40
MIN_SENTENCE_LENGTH = 12
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 0.20
MAX_RETRIES = 4
DEFAULT_INPUT_FILE = "claims_100_labeled.csv"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "FactCheckingStudentProject/3.0 "
            "(educational use; contact: student-project@example.com)"
        )
    }
)

LAST_REQUEST_TIME = 0.0


def softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    return set(normalize_text(text).split())


def extract_main_subject(claim: str) -> Optional[str]:
    claim = claim.strip()
    claim = re.sub(r"^the\s+", "", claim, flags=re.IGNORECASE)

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
        r"\s+located\s+",
        r"\s+flows\s+",
        r"\s+wrote\s+",
        r"\s+painted\s+",
        r"\s+composed\s+",
        r"\s+created\s+",
        r"\s+invented\s+",
        r"\s+sank\s+",
        r"\s+ended\s+",
    ]

    match = re.search("|".join(verbs), claim, flags=re.IGNORECASE)

    if match:
        subject = claim[:match.start()].strip()
    else:
        subject = " ".join(claim.split()[:5])

    subject = re.sub(r"[^A-Za-z0-9\s\-']", " ", subject)
    subject = " ".join(subject.split())

    return subject if subject else None


def sentence_relevance(sentence: str, claim: str, subject: str) -> float:
    sentence_tokens = tokenize(sentence)
    claim_tokens = tokenize(claim)
    subject_tokens = tokenize(subject)

    claim_overlap = len(sentence_tokens & claim_tokens)
    subject_overlap = len(sentence_tokens & subject_tokens)

    score = claim_overlap + 2.0 * subject_overlap

    normalized_sentence = normalize_text(sentence)
    normalized_subject = normalize_text(subject)

    if normalized_subject and normalized_subject in normalized_sentence:
        score += 4.0

    return score


def wikipedia_get(params: dict) -> Optional[dict]:
    global LAST_REQUEST_TIME

    api_url = "https://en.wikipedia.org/w/api.php"
    params = dict(params)
    params["maxlag"] = 5

    for attempt in range(MAX_RETRIES):
        elapsed = time.monotonic() - LAST_REQUEST_TIME

        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

        try:
            response = SESSION.get(
                api_url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            LAST_REQUEST_TIME = time.monotonic()

            if response.status_code in (429, 500, 502, 503, 504):
                wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                print(
                    f"Wikipedia temporaneamente non disponibile "
                    f"(HTTP {response.status_code}). Attendo {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            error_code = data.get("error", {}).get("code")

            if error_code in ("ratelimited", "maxlag"):
                wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                print(
                    f"Wikipedia richiede di rallentare ({error_code}). "
                    f"Attendo {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue

            return data

        except (requests.RequestException, ValueError) as error:
            if attempt == MAX_RETRIES - 1:
                print(f"Richiesta Wikipedia fallita dopo {MAX_RETRIES} tentativi: {error}")
                return None

            wait_time = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Errore di rete: {error}. Nuovo tentativo tra {wait_time:.1f}s...")
            time.sleep(wait_time)

    return None


def fetch_wikipedia_evidence(claim: str, verbose: bool = True) -> list[tuple[str, str]]:
    subject = extract_main_subject(claim)

    if not subject:
        return []

    search_data = wikipedia_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": subject,
            "format": "json",
            "srlimit": MAX_WIKIPEDIA_PAGES,
            "utf8": 1,
        }
    )

    if not search_data:
        return []

    search_results = search_data.get("query", {}).get("search", [])

    if not search_results:
        return []

    titles = [item["title"] for item in search_results]
    normalized_subject = normalize_text(subject)

    exact_title = next(
        (title for title in titles if normalize_text(title) == normalized_subject),
        None,
    )

    if exact_title:
        selected_titles = [exact_title] + [
            title for title in titles if title != exact_title
        ]
    else:
        selected_titles = titles

    selected_titles = selected_titles[:MAX_WIKIPEDIA_PAGES]

    if verbose:
        print(f"Soggetto estratto: {subject}")
        print("Titoli trovati:", ", ".join(selected_titles))

    # Recupera le introduzioni di tutte le pagine scelte in UNA sola richiesta.
    extract_data = wikipedia_get(
        {
            "action": "query",
            "format": "json",
            "titles": "|".join(selected_titles),
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }
    )

    if not extract_data:
        return []

    pages = extract_data.get("query", {}).get("pages", {})
    evidence = []

    for page in pages.values():
        extract = page.get("extract", "").strip()

        if extract:
            evidence.append((page.get("title", ""), extract))

    return evidence


def get_candidate_sentences(
    evidence: list[tuple[str, str]],
    claim: str,
    subject: str,
) -> list[tuple[str, str]]:
    ranked = []

    for title, excerpt in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+", excerpt):
            sentence = sentence.strip()

            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            score = sentence_relevance(sentence, claim, subject)
            ranked.append((score, title, sentence))

    ranked.sort(key=lambda item: item[0], reverse=True)

    candidates = []
    seen = set()

    for _, title, sentence in ranked:
        normalized_sentence = normalize_text(sentence)

        if normalized_sentence in seen:
            continue

        seen.add(normalized_sentence)
        candidates.append((title, sentence))

        if len(candidates) >= MAX_SENTENCES_PER_CLAIM:
            break

    return candidates


def load_model() -> CrossEncoder:
    print("\nCaricamento del modello NLI...")
    print("Il modello viene caricato una volta sola e riutilizzato.")
    return CrossEncoder(MODEL_NAME)


def empty_result(claim: str) -> dict:
    return {
        "claim": claim,
        "subject": extract_main_subject(claim) or "",
        "predicted_label": "NOT ENOUGH INFO",
        "confidence": 0.0,
        "contradiction": 0.0,
        "entailment": 0.0,
        "neutral": 1.0,
        "page": "",
        "evidence": "",
        "candidate_sentences": 0,
    }


def verify_claim(claim: str, model: CrossEncoder, verbose: bool = True) -> dict:
    result = empty_result(claim)
    subject = result["subject"]

    if verbose:
        print("\n" + "=" * 72)
        print(f"CLAIM: {claim}")
        print("=" * 72)
        print("Ricerca evidenza su Wikipedia...")

    if not subject:
        if verbose:
            print("Soggetto non identificabile.")
        return result

    evidence = fetch_wikipedia_evidence(claim, verbose=verbose)

    if not evidence:
        if verbose:
            print("Nessuna evidenza Wikipedia recuperata.")
        return result

    candidates = get_candidate_sentences(evidence, claim, subject)
    result["candidate_sentences"] = len(candidates)

    if not candidates:
        if verbose:
            print("Nessuna frase candidata disponibile.")
        return result

    pairs = [(sentence, claim) for _, sentence in candidates]
    logits_batch = model.predict(pairs, batch_size=16, show_progress_bar=False)

    best_support = None
    best_refute = None

    for (title, sentence), logits in zip(candidates, logits_batch):
        probabilities = softmax(logits)
        candidate = {
            "page": title,
            "evidence": sentence,
            "contradiction": float(probabilities[0]),
            "entailment": float(probabilities[1]),
            "neutral": float(probabilities[2]),
        }

        if best_support is None or candidate["entailment"] > best_support["entailment"]:
            best_support = candidate

        if best_refute is None or candidate["contradiction"] > best_refute["contradiction"]:
            best_refute = candidate

    support_score = best_support["entailment"]
    refute_score = best_refute["contradiction"]

    # Prima controlla se esiste almeno una evidenza forte per SUPPORTS o REFUTES.
    # Se entrambe esistono, sceglie quella con il punteggio più alto.
    if support_score >= ENTAILMENT_THRESHOLD and refute_score >= CONTRADICTION_THRESHOLD:
        if support_score >= refute_score:
            verdict = "SUPPORTS"
            chosen = best_support
            confidence = support_score
        else:
            verdict = "REFUTES"
            chosen = best_refute
            confidence = refute_score

    elif support_score >= ENTAILMENT_THRESHOLD:
        verdict = "SUPPORTS"
        chosen = best_support
        confidence = support_score

    elif refute_score >= CONTRADICTION_THRESHOLD:
        verdict = "REFUTES"
        chosen = best_refute
        confidence = refute_score

    else:
        verdict = "NOT ENOUGH INFO"
        chosen = max(
            (best_support, best_refute),
            key=lambda candidate: candidate["neutral"],
        )
        confidence = chosen["neutral"]

    result.update(
        {
            "predicted_label": verdict,
            "confidence": confidence,
            "contradiction": chosen["contradiction"],
            "entailment": chosen["entailment"],
            "neutral": chosen["neutral"],
            "page": chosen["page"],
            "evidence": chosen["evidence"],
        }
    )

    if verbose:
        print(f"Pagine recuperate: {len(evidence)}")
        print(f"Frasi candidate analizzate: {len(candidates)}")
        print("\nEVIDENZA SELEZIONATA:")
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

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "claim" not in reader.fieldnames:
            raise ValueError("Il CSV deve contenere una colonna chiamata 'claim'.")

        for index, row in enumerate(reader, start=1):
            claim = row.get("claim", "").strip()

            if not claim:
                continue

            gold_label = row.get("gold_label", "").strip().upper()

            if gold_label not in LABELS:
                gold_label = ""

            claims.append(
                {
                    "id": row.get("id", str(index)).strip() or str(index),
                    "claim": claim,
                    "gold_label": gold_label,
                }
            )

    return claims


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def build_statistics(results: list[dict], input_file: str, elapsed_seconds: float) -> dict:
    labeled_results = [row for row in results if row["gold_label"] in LABELS]
    predictions = Counter(row["predicted_label"] for row in results)
    gold_labels = Counter(row["gold_label"] for row in labeled_results)

    confusion_matrix = {
        gold: {predicted: 0 for predicted in LABELS}
        for gold in LABELS
    }

    for row in labeled_results:
        confusion_matrix[row["gold_label"]][row["predicted_label"]] += 1

    correct_predictions = sum(
        row["gold_label"] == row["predicted_label"]
        for row in labeled_results
    )

    per_label = {}

    for label in LABELS:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(
            confusion_matrix[gold][label]
            for gold in LABELS
            if gold != label
        )
        false_negative = sum(
            confusion_matrix[label][predicted]
            for predicted in LABELS
            if predicted != label
        )

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gold_labels[label],
        }

    macro_f1 = safe_divide(
        sum(per_label[label]["f1"] for label in LABELS),
        len(LABELS),
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": input_file,
        "total_claims": len(results),
        "labeled_claims": len(labeled_results),
        "correct_predictions": correct_predictions,
        "accuracy": safe_divide(correct_predictions, len(labeled_results)),
        "macro_f1": macro_f1,
        "average_confidence": safe_divide(
            sum(float(row["confidence"]) for row in results),
            len(results),
        ),
        "elapsed_seconds": elapsed_seconds,
        "predicted_distribution": dict(predictions),
        "gold_distribution": dict(gold_labels),
        "confusion_matrix": confusion_matrix,
        "per_label": per_label,
    }


def write_results_csv(results: list[dict], output_path: Path) -> None:
    fields = [
        "id",
        "claim",
        "gold_label",
        "predicted_label",
        "is_correct",
        "confidence",
        "contradiction",
        "entailment",
        "neutral",
        "subject",
        "candidate_sentences",
        "page",
        "evidence",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def write_statistics_files(statistics: dict, output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "statistics.json"
    report_path = output_dir / "statistics_report.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2, ensure_ascii=False)

    lines = [
        "STATISTICHE FACT-CHECKING",
        "=" * 72,
        f"Generato il: {statistics['generated_at']}",
        f"File input: {statistics['input_file']}",
        f"Claim totali: {statistics['total_claims']}",
        f"Claim etichettate: {statistics['labeled_claims']}",
        f"Predizioni corrette: {statistics['correct_predictions']}",
        f"Accuracy: {statistics['accuracy']:.2%}",
        f"Macro F1: {statistics['macro_f1']:.4f}",
        f"Confidenza media: {statistics['average_confidence']:.4f}",
        f"Tempo totale: {statistics['elapsed_seconds']:.1f} secondi",
        "",
        "DISTRIBUZIONE ETICHETTE",
        "-" * 72,
    ]

    for label in LABELS:
        lines.append(
            f"{label}: gold={statistics['gold_distribution'].get(label, 0)}, "
            f"predette={statistics['predicted_distribution'].get(label, 0)}"
        )

    lines.extend(
        [
            "",
            "MATRICE DI CONFUSIONE",
            "Righe = etichetta reale; colonne = predizione",
            "-" * 72,
            f"{'GOLD \\ PRED':<22}{'SUPPORTS':>16}{'REFUTES':>16}{'NOT ENOUGH INFO':>20}",
        ]
    )

    for gold in LABELS:
        row = statistics["confusion_matrix"][gold]
        lines.append(
            f"{gold:<22}{row['SUPPORTS']:>16}{row['REFUTES']:>16}"
            f"{row['NOT ENOUGH INFO']:>20}"
        )

    lines.extend(
        [
            "",
            "METRICHE PER CLASSE",
            "-" * 72,
            f"{'Classe':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}",
        ]
    )

    for label in LABELS:
        metrics = statistics["per_label"][label]
        lines.append(
            f"{label:<22}{metrics['precision']:>12.4f}{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}{metrics['support']:>12}"
        )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    return json_path, report_path


def create_output_directory() -> Path:
    folder = Path("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def verify_single_claim(model: CrossEncoder) -> None:
    print("\n" + "=" * 72)
    print("VERIFICA DI UNA SINGOLA CLAIM")
    print("=" * 72)

    claim = input("Scrivi una claim in inglese: ").strip()

    if not claim:
        print("La claim non può essere vuota.")
        return

    start_time = time.perf_counter()
    prediction = verify_claim(claim, model, verbose=True)
    elapsed_seconds = time.perf_counter() - start_time

    output_dir = create_output_directory()
    output_row = {
        "id": "1",
        "claim": claim,
        "gold_label": "",
        "predicted_label": prediction["predicted_label"],
        "is_correct": "",
        "confidence": round(prediction["confidence"], 4),
        "contradiction": round(prediction["contradiction"], 4),
        "entailment": round(prediction["entailment"], 4),
        "neutral": round(prediction["neutral"], 4),
        "subject": prediction["subject"],
        "candidate_sentences": prediction["candidate_sentences"],
        "page": prediction["page"],
        "evidence": prediction["evidence"],
    }

    write_results_csv([output_row], output_dir / "results.csv")
    statistics = build_statistics([output_row], "manual_claim", elapsed_seconds)
    _, report_path = write_statistics_files(statistics, output_dir)

    print(f"\nRisultato salvato in: {output_dir / 'results.csv'}")
    print(f"Statistiche salvate in: {report_path}")


def verify_claims_from_file(model: CrossEncoder) -> None:
    print("\n" + "=" * 72)
    print("VERIFICA DI CLAIM DA FILE CSV")
    print("=" * 72)

    filename = input(
        f"Nome file CSV [Invio = {DEFAULT_INPUT_FILE}]: "
    ).strip() or DEFAULT_INPUT_FILE

    input_path = Path(filename)

    if not input_path.exists():
        print(f"Il file '{filename}' non esiste nella cartella corrente.")
        return

    try:
        claims = read_claims_from_csv(str(input_path))
    except (OSError, ValueError) as error:
        print(f"Errore nella lettura del CSV: {error}")
        return

    if not claims:
        print("Il file non contiene claim valide.")
        return

    output_dir = create_output_directory()
    results = []
    start_time = time.perf_counter()

    print(f"\nClaim da elaborare: {len(claims)}")
    print(
        "Le richieste a Wikipedia sono seriali, rallentate e con retry automatico "
        "per rispettare il servizio.\n"
    )

    for index, item in enumerate(claims, start=1):
        print(f"[{index}/{len(claims)}] {item['claim']}")
        prediction = verify_claim(item["claim"], model, verbose=False)

        is_correct = ""
        if item["gold_label"]:
            is_correct = prediction["predicted_label"] == item["gold_label"]

        results.append(
            {
                "id": item["id"],
                "claim": item["claim"],
                "gold_label": item["gold_label"],
                "predicted_label": prediction["predicted_label"],
                "is_correct": is_correct,
                "confidence": round(prediction["confidence"], 4),
                "contradiction": round(prediction["contradiction"], 4),
                "entailment": round(prediction["entailment"], 4),
                "neutral": round(prediction["neutral"], 4),
                "subject": prediction["subject"],
                "candidate_sentences": prediction["candidate_sentences"],
                "page": prediction["page"],
                "evidence": prediction["evidence"],
            }
        )

    elapsed_seconds = time.perf_counter() - start_time
    write_results_csv(results, output_dir / "results.csv")
    statistics = build_statistics(results, str(input_path), elapsed_seconds)
    _, report_path = write_statistics_files(statistics, output_dir)

    print("\n" + "=" * 72)
    print("ELABORAZIONE COMPLETATA")
    print("=" * 72)
    print(f"Cartella risultati: {output_dir}")
    print(f"Accuracy: {statistics['accuracy']:.2%}")
    print(f"Macro F1: {statistics['macro_f1']:.4f}")
    print(f"Tempo totale: {elapsed_seconds:.1f} secondi")
    print(f"Report statistiche: {report_path}")


def print_menu() -> None:
    print("\n" + "=" * 72)
    print("SISTEMA DI FACT-CHECKING CON WIKIPEDIA E NLI")
    print("=" * 72)
    print("1 - Verificare una claim scritta a mano")
    print("2 - Verificare tutte le claim di un file CSV")
    print("3 - Uscire")


def main() -> None:
    model = load_model()

    while True:
        print_menu()
        choice = input("Scegli un'opzione (1, 2, 3): ").strip()

        if choice == "1":
            verify_single_claim(model)
        elif choice == "2":
            verify_claims_from_file(model)
        elif choice == "3":
            print("Chiusura del programma.")
            sys.exit(0)
        else:
            print("Scelta non valida. Inserisci 1, 2 oppure 3.")


if __name__ == "__main__":
    main()

import csv
import json
import math
import os
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
ENTAILMENT_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.55
MAX_WIKIPEDIA_PAGES = 5
MAX_SENTENCES_PER_CLAIM = 40
MIN_SENTENCE_LENGTH = 12
DEFAULT_INPUT_FILE = "claims_100_labeled.csv"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "FactCheckingStudentProject/2.0 "
            "(educational project; contact@example.com)"
        )
    }
)


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


def sentence_score(sentence: str, claim: str, subject: str) -> float:
    sentence_tokens = tokenize(sentence)
    claim_tokens = tokenize(claim)
    subject_tokens = tokenize(subject)

    overlap = len(sentence_tokens & claim_tokens)
    subject_overlap = len(sentence_tokens & subject_tokens)

    score = overlap + 2.0 * subject_overlap

    normalized_subject = normalize_text(subject)
    normalized_sentence = normalize_text(sentence)

    if normalized_subject and normalized_subject in normalized_sentence:
        score += 4.0

    return score


def fetch_wikipedia_evidence(claim: str, verbose: bool = True) -> list[tuple[str, str]]:
    subject = extract_main_subject(claim)

    if not subject:
        return []

    wikipedia_api = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": subject,
        "format": "json",
        "srlimit": MAX_WIKIPEDIA_PAGES,
        "utf8": 1,
    }

    try:
        response = SESSION.get(wikipedia_api, params=search_params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        if verbose:
            print(f"Errore nella ricerca Wikipedia: {error}")
        return []

    search_results = data.get("query", {}).get("search", [])

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
            response = SESSION.get(wikipedia_api, params=extract_params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            continue

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)

        if not page:
            continue

        extract = page.get("extract", "").strip()

        if extract:
            evidence.append((page.get("title", title), extract))

    return evidence


def get_candidate_sentences(
    evidence: list[tuple[str, str]],
    claim: str,
    subject: str,
) -> list[tuple[str, str]]:
    candidates = []

    for title, excerpt in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", excerpt)

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            score = sentence_score(sentence, claim, subject)
            candidates.append((score, title, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)

    unique = []
    seen = set()

    for _, title, sentence in candidates:
        key = normalize_text(sentence)

        if key not in seen:
            seen.add(key)
            unique.append((title, sentence))

        if len(unique) >= MAX_SENTENCES_PER_CLAIM:
            break

    return unique


def load_model() -> CrossEncoder:
    print("\nCaricamento del modello NLI...")
    print("Il modello viene mantenuto in memoria per tutte le claim.")
    return CrossEncoder(MODEL_NAME)


def empty_result(claim: str) -> dict:
    return {
        "claim": claim,
        "predicted_label": "NOT ENOUGH INFO",
        "confidence": 0.0,
        "contradiction": 0.0,
        "entailment": 0.0,
        "neutral": 1.0,
        "page": "",
        "evidence": "",
        "subject": extract_main_subject(claim) or "",
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
            print("Nessuna pagina Wikipedia recuperata.")
        return result

    candidates = get_candidate_sentences(evidence, claim, subject)
    result["candidate_sentences"] = len(candidates)

    if not candidates:
        if verbose:
            print("Nessuna frase candidata trovata.")
        return result

    pairs = [(sentence, claim) for _, sentence in candidates]
    logits_batch = model.predict(pairs, batch_size=16, show_progress_bar=False)

    best_support = None
    best_refute = None

    for (title, sentence), logits in zip(candidates, logits_batch):
        probabilities = softmax(logits)
        contradiction = float(probabilities[0])
        entailment = float(probabilities[1])
        neutral = float(probabilities[2])

        candidate = {
            "page": title,
            "evidence": sentence,
            "contradiction": contradiction,
            "entailment": entailment,
            "neutral": neutral,
        }

        if best_support is None or entailment > best_support["entailment"]:
            best_support = candidate

        if best_refute is None or contradiction > best_refute["contradiction"]:
            best_refute = candidate

    support_score = best_support["entailment"]
    refute_score = best_refute["contradiction"]

    if support_score >= ENTAILMENT_THRESHOLD and support_score >= refute_score:
        verdict = "SUPPORTS"
        chosen = best_support
        confidence = support_score
    elif refute_score >= CONTRADICTION_THRESHOLD and refute_score > support_score:
        verdict = "REFUTES"
        chosen = best_refute
        confidence = refute_score
    else:
        verdict = "NOT ENOUGH INFO"
        chosen = best_support
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
    rows = []

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "claim" not in reader.fieldnames:
            raise ValueError("Il CSV deve contenere la colonna 'claim'.")

        for index, row in enumerate(reader, start=1):
            claim = row.get("claim", "").strip()

            if not claim:
                continue

            gold_label = row.get("gold_label", "").strip().upper()

            if gold_label and gold_label not in LABELS:
                gold_label = ""

            rows.append(
                {
                    "id": row.get("id", str(index)).strip() or str(index),
                    "claim": claim,
                    "gold_label": gold_label,
                }
            )

    return rows


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_statistics(results: list[dict], input_file: str, elapsed_seconds: float) -> dict:
    labeled = [row for row in results if row["gold_label"] in LABELS]
    predictions = Counter(row["predicted_label"] for row in results)
    gold_distribution = Counter(row["gold_label"] for row in labeled)

    correct = sum(
        row["predicted_label"] == row["gold_label"]
        for row in labeled
    )

    confusion = {
        gold: {predicted: 0 for predicted in LABELS}
        for gold in LABELS
    }

    for row in labeled:
        confusion[row["gold_label"]][row["predicted_label"]] += 1

    per_label = {}

    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[gold][label]
            for gold in LABELS
            if gold != label
        )
        false_negative = sum(
            confusion[label][predicted]
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
            "support": gold_distribution[label],
        }

    macro_f1 = safe_divide(
        sum(per_label[label]["f1"] for label in LABELS),
        len(LABELS),
    )

    average_confidence = safe_divide(
        sum(float(row["confidence"]) for row in results),
        len(results),
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": input_file,
        "total_claims": len(results),
        "labeled_claims": len(labeled),
        "correct_predictions": correct,
        "accuracy": safe_divide(correct, len(labeled)),
        "macro_f1": macro_f1,
        "average_confidence": average_confidence,
        "elapsed_seconds": elapsed_seconds,
        "predicted_distribution": dict(predictions),
        "gold_distribution": dict(gold_distribution),
        "confusion_matrix": confusion,
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

    lines = []
    lines.append("STATISTICHE FACT-CHECKING")
    lines.append("=" * 72)
    lines.append(f"Generato il: {statistics['generated_at']}")
    lines.append(f"File input: {statistics['input_file']}")
    lines.append(f"Claim totali: {statistics['total_claims']}")
    lines.append(f"Claim etichettate: {statistics['labeled_claims']}")
    lines.append(f"Predizioni corrette: {statistics['correct_predictions']}")
    lines.append(f"Accuracy: {statistics['accuracy']:.2%}")
    lines.append(f"Macro F1: {statistics['macro_f1']:.4f}")
    lines.append(f"Confidenza media: {statistics['average_confidence']:.4f}")
    lines.append(f"Tempo totale: {statistics['elapsed_seconds']:.1f} secondi")
    lines.append("")
    lines.append("DISTRIBUZIONE ETICHETTE")
    lines.append("-" * 72)

    for label in LABELS:
        lines.append(
            f"{label}: gold={statistics['gold_distribution'].get(label, 0)}, "
            f"predette={statistics['predicted_distribution'].get(label, 0)}"
        )

    lines.append("")
    lines.append("MATRICE DI CONFUSIONE")
    lines.append("Righe = etichetta reale; colonne = predizione")
    lines.append("-" * 72)
    lines.append(
        f"{'GOLD \\ PRED':<22}"
        f"{'SUPPORTS':>16}{'REFUTES':>16}{'NOT ENOUGH INFO':>20}"
    )

    for gold in LABELS:
        row = statistics["confusion_matrix"][gold]
        lines.append(
            f"{gold:<22}"
            f"{row['SUPPORTS']:>16}"
            f"{row['REFUTES']:>16}"
            f"{row['NOT ENOUGH INFO']:>20}"
        )

    lines.append("")
    lines.append("METRICHE PER CLASSE")
    lines.append("-" * 72)
    lines.append(
        f"{'Classe':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}"
    )

    for label in LABELS:
        metrics = statistics["per_label"][label]
        lines.append(
            f"{label:<22}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['support']:>12}"
        )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    return json_path, report_path


def choose_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def verify_single_claim(model: CrossEncoder) -> None:
    print("\n" + "=" * 72)
    print("VERIFICA DI UNA SINGOLA CLAIM")
    print("=" * 72)

    claim = input("Scrivi una claim in inglese: ").strip()

    if not claim:
        print("La claim non può essere vuota.")
        return

    start = time.perf_counter()
    result = verify_claim(claim, model, verbose=True)
    elapsed = time.perf_counter() - start

    output_dir = choose_output_dir()
    row = {
        "id": "1",
        "claim": claim,
        "gold_label": "",
        "predicted_label": result["predicted_label"],
        "is_correct": "",
        "confidence": round(result["confidence"], 4),
        "contradiction": round(result["contradiction"], 4),
        "entailment": round(result["entailment"], 4),
        "neutral": round(result["neutral"], 4),
        "subject": result["subject"],
        "candidate_sentences": result["candidate_sentences"],
        "page": result["page"],
        "evidence": result["evidence"],
    }

    write_results_csv([row], output_dir / "results.csv")
    statistics = build_statistics([row], "manual_claim", elapsed)
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

    output_dir = choose_output_dir()
    results = []
    start = time.perf_counter()

    print(f"\nClaim da elaborare: {len(claims)}")
    print("I risultati vengono salvati alla fine della procedura.\n")

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

    elapsed = time.perf_counter() - start
    write_results_csv(results, output_dir / "results.csv")
    statistics = build_statistics(results, str(input_path), elapsed)
    _, report_path = write_statistics_files(statistics, output_dir)

    print("\n" + "=" * 72)
    print("ELABORAZIONE COMPLETATA")
    print("=" * 72)
    print(f"Cartella risultati: {output_dir}")
    print(f"Accuracy: {statistics['accuracy']:.2%}")
    print(f"Macro F1: {statistics['macro_f1']:.4f}")
    print(f"Tempo totale: {elapsed:.1f} secondi")
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
            print("Scelta non valida.")


if __name__ == "__main__":
    main()

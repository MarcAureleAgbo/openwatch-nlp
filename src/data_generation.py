"""Generate a synthetic text classification dataset for OpenWatch NLP."""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "raw" / "synthetic_public_signals.csv"
DEFAULT_RANDOM_SEED = 42


RED_PATTERNS = [
    "Public post mentions leaked {asset} credentials and active exploitation attempts.",
    "Several users claim that {asset} access was compromised after a suspicious reset campaign.",
    "A forum thread shares administrator passwords related to {asset} and asks others to test them.",
    "Multiple reports describe unauthorized withdrawals after account takeover on {asset}.",
    "A public message threatens to publish private customer data from {asset} tonight.",
    "Users report confirmed malware links sent through official-looking {asset} support messages.",
    "A paste site appears to expose internal tokens connected to {asset} environments.",
    "Community moderators warn that attackers are actively bypassing two-factor checks on {asset}.",
]

ORANGE_PATTERNS = [
    "Several users report unusual login prompts when trying to access {asset}.",
    "A growing thread complains about delayed account verification on {asset}.",
    "Some customers mention suspicious emails that look similar to {asset} notifications.",
    "Users describe repeated password reset messages, but no confirmed breach is reported.",
    "A support forum shows a sudden increase in complaints about blocked accounts on {asset}.",
    "A public review mentions unexpected payment declines and asks whether others saw the same issue.",
    "Posts about {asset} show rising frustration after unexplained session timeouts.",
    "A community member asks if recent security warnings from {asset} are legitimate.",
]

GREEN_PATTERNS = [
    "The company announced a new interface update for {asset} with minor usability changes.",
    "A user asks how to change notification settings in the {asset} mobile app.",
    "The weekly newsletter highlights upcoming product webinars and customer stories.",
    "A public article compares pricing options for several productivity platforms.",
    "Users discuss preferred dashboard layouts and reporting exports in {asset}.",
    "The release notes describe visual improvements and accessibility fixes.",
    "A customer shares a positive onboarding experience with the {asset} support team.",
    "A blog post explains basic account settings and profile customization.",
]

ASSETS = [
    "Northstar",
    "CloudDesk",
    "PayLink",
    "SecureHub",
    "MarketBoard",
    "OpenGate",
    "BrightPanel",
    "DataPilot",
]

SOURCE_TYPES = ["forum", "public_post", "review", "news", "support_thread"]


def _generate_records_for_label(
    label: str,
    patterns: list[str],
    count: int,
    start_id: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    records = []
    start_date = date(2026, 1, 1)

    for index in range(count):
        pattern = rng.choice(patterns)
        asset = rng.choice(ASSETS)
        created_at = start_date + timedelta(days=rng.randint(0, 120))

        records.append(
            {
                "text_id": f"{label}-{start_id + index:04d}",
                "text": pattern.format(asset=asset),
                "label": label,
                "source_type": rng.choice(SOURCE_TYPES),
                "created_at": created_at.isoformat(),
            }
        )

    return records


def generate_dataset(records_per_class: int = 200, random_seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    """Generate a balanced synthetic dataset for the three risk labels."""
    rng = random.Random(random_seed)
    records: list[dict[str, str]] = []

    records.extend(_generate_records_for_label("red", RED_PATTERNS, records_per_class, 1, rng))
    records.extend(_generate_records_for_label("orange", ORANGE_PATTERNS, records_per_class, 1, rng))
    records.extend(_generate_records_for_label("green", GREEN_PATTERNS, records_per_class, 1, rng))

    rng.shuffle(records)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OpenWatch NLP data.")
    parser.add_argument("--records-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    dataset = generate_dataset(
        records_per_class=args.records_per_class,
        random_seed=args.random_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"Generated {len(dataset)} rows at {args.output}")
    print(dataset["label"].value_counts().to_string())


if __name__ == "__main__":
    main()

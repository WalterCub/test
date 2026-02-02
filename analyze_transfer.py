#!/usr/bin/env python3
"""Analyze technology transfer event attendance with fuzzy deduplication."""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


try:
    from rapidfuzz.distance import Levenshtein

    def similarity(a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        if a == b:
            return 100.0
        return 100.0 * (1.0 - Levenshtein.normalized_distance(a, b))

except Exception:

    def _levenshtein_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        if len(a) < len(b):
            a, b = b, a

        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current_row = [i]
            for j, cb in enumerate(b, start=1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def similarity(a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        if a == b:
            return 100.0
        distance = _levenshtein_distance(a, b)
        return 100.0 * (1.0 - distance / max(len(a), len(b)))


NAME_WEIGHT = 0.15
ID_WEIGHT = 0.60
PHONE_WEIGHT = 0.25
MATCH_THRESHOLD = 80.0


@dataclass
class UnionFind:
    parent: Dict[int, int]
    rank: Dict[int, int]

    @classmethod
    def from_items(cls, items: Iterable[int]) -> "UnionFind":
        parent = {item: item for item in items}
        rank = {item: 0 for item in items}
        return cls(parent=parent, rank=rank)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "").isdigit():
        text = text[:-2]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = " ".join(text.split())
    return text.upper()


def normalize_digits(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "").isdigit():
        text = text[:-2]
    digits = "".join(char for char in text if char.isdigit())
    return digits


def compute_match_score(row_a: pd.Series, row_b: pd.Series) -> float:
    id_score = similarity(row_a["CEDULA_NORM"], row_b["CEDULA_NORM"])
    name_score = similarity(row_a["NOMBRE_NORM"], row_b["NOMBRE_NORM"])
    phone_score = similarity(row_a["CELULAR_NORM"], row_b["CELULAR_NORM"])
    return (
        ID_WEIGHT * id_score
        + NAME_WEIGHT * name_score
        + PHONE_WEIGHT * phone_score
    )


def blocking_key(row: pd.Series) -> str:
    cedula = row["CEDULA_NORM"]
    celular = row["CELULAR_NORM"]
    nombre = row["NOMBRE_NORM"]
    if len(cedula) >= 4:
        return f"CED:{cedula[-4:]}"
    if len(celular) >= 4:
        return f"CEL:{celular[-4:]}"
    if nombre:
        return f"NOM:{nombre[:3]}"
    return "NOM:UNK"


def cluster_dataframe(
    df: pd.DataFrame,
    prefix: str,
    threshold: float = MATCH_THRESHOLD,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)

    uf = UnionFind.from_items(df.index)
    blocks = df.groupby("BLOCK_KEY").groups

    for _, indices in blocks.items():
        idx_list = list(indices)
        if len(idx_list) < 2:
            continue
        for i, idx_a in enumerate(idx_list[:-1]):
            row_a = df.loc[idx_a]
            for idx_b in idx_list[i + 1 :]:
                row_b = df.loc[idx_b]
                score = compute_match_score(row_a, row_b)
                if score >= threshold:
                    uf.union(idx_a, idx_b)

    clusters: Dict[int, str] = {}
    cluster_ids: Dict[int, str] = {}
    counter = 1
    for idx in df.index:
        root = uf.find(idx)
        if root not in clusters:
            clusters[root] = f"{prefix}{counter}"
            counter += 1
        cluster_ids[idx] = clusters[root]
    return pd.Series(cluster_ids)


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["NOMBRE_NORM"] = normalized["NOMBRE"].apply(normalize_text)
    normalized["CEDULA_NORM"] = normalized["CEDULA"].apply(normalize_digits)
    normalized["CELULAR_NORM"] = normalized["CELULAR"].apply(normalize_digits)
    normalized["TIPO_PARTICIPANTE"] = normalized["TIPO_PARTICIPANTE"].fillna("OTRO")
    normalized["TIPO_PARTICIPANTE"] = normalized["TIPO_PARTICIPANTE"].apply(
        lambda value: normalize_text(value) or "OTRO"
    )
    normalized["BLOCK_KEY"] = normalized.apply(blocking_key, axis=1)
    return normalized


def load_inputs(events_path: str, attendees_path: str) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    attendees = pd.read_csv(attendees_path)

    events.columns = [col.strip().upper() for col in events.columns]
    attendees.columns = [col.strip().upper() for col in attendees.columns]

    required_events = {"ZONA", "SECCIONAL", "INGENIERO", "ID_EVENTO", "ID_EXCEL"}
    missing_events = required_events - set(events.columns)
    if missing_events:
        raise ValueError(f"Faltan columnas en eventos: {sorted(missing_events)}")

    required_attendees = {"NOMBRE", "CEDULA", "CELULAR", "TIPO_PARTICIPANTE", "ID_EXCEL"}
    missing_attendees = required_attendees - set(attendees.columns)
    if missing_attendees:
        raise ValueError(f"Faltan columnas en asistentes: {sorted(missing_attendees)}")

    merged = attendees.merge(
        events,
        on="ID_EXCEL",
        how="inner",
        suffixes=("", "_EVENT"),
    )
    return merged


def cluster_by_group(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.Series:
    cluster_series = pd.Series(index=df.index, dtype=str)
    for group_value, group_df in df.groupby(group_col):
        clusters = cluster_dataframe(group_df, prefix=f"{prefix}{group_value}_")
        cluster_series.loc[group_df.index] = clusters
    return cluster_series


def summarize_engineers(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    type_values = ["AGRICULTOR", "ASISTENTE TECNICO", "OTRO"]

    for engineer, engineer_df in df.groupby("INGENIERO"):
        row: Dict[str, object] = {
            "INGENIERO": engineer,
            "ZONA": engineer_df["ZONA"].iloc[0],
            "SECCIONAL": engineer_df["SECCIONAL"].iloc[0],
            "UNIQUE_INTERNO": engineer_df["CLUSTER_ING"].nunique(),
            "UNIQUE_EN_ZONA": engineer_df["CLUSTER_ZONA"].nunique(),
            "UNIQUE_EN_SECCIONAL": engineer_df["CLUSTER_SECCIONAL"].nunique(),
        }

        for tipo in type_values:
            tipo_df = engineer_df[engineer_df["TIPO_PARTICIPANTE"] == tipo]
            row[f"UNIQUE_INTERNO_{tipo}"] = tipo_df["CLUSTER_ING"].nunique()
            row[f"UNIQUE_EN_ZONA_{tipo}"] = tipo_df["CLUSTER_ZONA"].nunique()
            row[f"UNIQUE_EN_SECCIONAL_{tipo}"] = tipo_df[
                "CLUSTER_SECCIONAL"
            ].nunique()

        records.append(row)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza asistencia a eventos de transferencia tecnológica."
    )
    parser.add_argument("--events", required=True, help="Ruta CSV de eventos")
    parser.add_argument("--attendees", required=True, help="Ruta CSV de asistentes")
    parser.add_argument("--output", required=True, help="Ruta CSV de salida")
    args = parser.parse_args()

    base = load_inputs(args.events, args.attendees)
    normalized = apply_normalization(base)

    normalized["CLUSTER_ING"] = cluster_by_group(
        normalized, "INGENIERO", prefix="ING_"
    )

    normalized["CLUSTER_ZONA"] = cluster_by_group(
        normalized, "ZONA", prefix="ZONA_"
    )

    normalized["CLUSTER_SECCIONAL"] = cluster_by_group(
        normalized, "SECCIONAL", prefix="SEC_"
    )

    summary = summarize_engineers(normalized)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

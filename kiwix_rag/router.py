from __future__ import annotations
import numpy as np


class GroupRouter:
    """
    Routes a query vector to the most relevant collection groups.

    Call build() once at startup with the list of available collection
    names and an initialized embedder. Then call route() per query.
    """

    def __init__(
        self,
        groups: dict,
        top_groups: int = 2,
        route_threshold: float = 0.20,
        max_per_group: int = 15,
    ) -> None:
        self._groups = groups
        self.top_groups = top_groups
        self.route_threshold = route_threshold
        self.max_per_group = max_per_group
        self.group_cols: dict[str, list[str]] = {}
        self._group_embs: dict[str, np.ndarray] = {}

    def build(self, available_names: list[str], embedder) -> None:
        """Assign collections to groups; embed group descriptions for routing."""
        assigned: set[str] = set()
        self.group_cols = {}

        for gname, gdef in self._groups.items():
            matched = [
                n for n in available_names
                if any(p in n for p in gdef["patterns"])
            ]
            if matched:
                self.group_cols[gname] = matched
                assigned.update(matched)

        unassigned = [n for n in available_names if n not in assigned]
        if unassigned:
            self.group_cols["_other"] = unassigned

        named = [g for g in self.group_cols if g != "_other"]
        if named:
            descs = [self._groups[g]["description"] for g in named]
            embs = embedder.encode(descs, normalize_embeddings=True)
            self._group_embs = {g: embs[i] for i, g in enumerate(named)}

    def route(self, query_vec: np.ndarray) -> list[str]:
        """Return group names most relevant to the normalized query vector."""
        if not self._group_embs:
            return list(self.group_cols.keys())

        scores = {
            g: float(np.dot(query_vec, emb))
            for g, emb in self._group_embs.items()
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best = ranked[0][1] if ranked else 0.0

        if best < self.route_threshold:
            fallback = [g for g, _ in ranked[: self.top_groups * 2]]
            if "_other" in self.group_cols:
                fallback.append("_other")
            return fallback

        # Confident match: search only the top groups. _other (unassigned
        # collections) is reserved for the below-threshold fallback above, so
        # it never pollutes correctly-routed queries.
        return [g for g, s in ranked[: self.top_groups] if s >= best - 0.1]

    def select_collections(
        self, names: list[str], query: str, max_n: int | None = None
    ) -> list[str]:
        """When a group has many collections, pick the most name-relevant ones."""
        cap = max_n if max_n is not None else self.max_per_group
        if len(names) <= cap:
            return names
        words = {w for w in query.lower().split() if len(w) > 3}
        def name_score(n: str) -> int:
            return sum(1 for w in words if w in n.lower())
        return sorted(names, key=name_score, reverse=True)[:cap]

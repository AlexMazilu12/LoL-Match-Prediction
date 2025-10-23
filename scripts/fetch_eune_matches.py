#Download EUNE Gold ranked matches (and timelines) for early-game feature work.

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Provide a clear message if requests is missing.
    import requests
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency 'requests'. Install it with 'pip install requests'.") from exc

LEAGUE_ROUTE = "https://eun1.api.riotgames.com"
MATCH_ROUTE = "https://europe.api.riotgames.com"
QUEUE = "RANKED_SOLO_5x5"
TIER = "GOLD"
DIVISIONS: Sequence[str] = ("I", "II", "III", "IV")
DEFAULT_QUEUE_ID = 420  # Ranked Solo/Duo
DEFAULT_TARGET_MATCHES = 1000
DEFAULT_MATCHES_PER_PLAYER = 50
DEFAULT_MIN_DURATION = 15 * 60  # seconds

logger = logging.getLogger("fetch_eune_matches")


def load_api_key(env_var: str = "RIOT_API_KEY", env_file: str = ".env") -> Optional[str]:
    """Load the Riot API key from the environment or a .env fallback."""

    key = os.getenv(env_var)
    if key:
        return key.strip()

    env_path = Path(env_file)
    if not env_path.is_file():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == env_var:
            return value.strip()
    return None


class RiotRateLimiter:
    """Simple client-side limiter aligned with developer key quotas (20 req/s, 100 req/2min)."""

    def __init__(self, per_second: int = 18, per_two_minutes: int = 95) -> None:
        self.per_second = per_second
        self.per_two_minutes = per_two_minutes
        self._second_window: deque[float] = deque()
        self._two_minute_window: deque[float] = deque()

    def wait(self) -> None:
        while True:
            now = time.time()
            self._trim_windows(now)
            if len(self._second_window) < self.per_second and len(self._two_minute_window) < self.per_two_minutes:
                self._second_window.append(now)
                self._two_minute_window.append(now)
                return
            sleep_for = self._calculate_sleep(now)
            time.sleep(sleep_for)

    def _trim_windows(self, now: float) -> None:
        while self._second_window and now - self._second_window[0] >= 1:
            self._second_window.popleft()
        while self._two_minute_window and now - self._two_minute_window[0] >= 120:
            self._two_minute_window.popleft()

    def _calculate_sleep(self, now: float) -> float:
        waits: List[float] = []
        if self._second_window:
            waits.append(max(0.05, 1 - (now - self._second_window[0])))
        if self._two_minute_window:
            waits.append(max(0.05, 120 - (now - self._two_minute_window[0])))
        return max(waits) if waits else 0.05


class RiotAPI:
    """Wrapper around requests.Session with retry + rate-limit logic."""

    def __init__(self, api_key: str, rate_limiter: Optional[RiotRateLimiter] = None, timeout: float = 10.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": api_key})
        self.timeout = timeout
        self.rate_limiter = rate_limiter or RiotRateLimiter()

    def get_json(self, url: str, params: Optional[Dict[str, object]] = None, retries: int = 5) -> Dict:
        for attempt in range(1, retries + 1):
            self.rate_limiter.wait()
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                logger.debug("Rate limited. Sleeping %.2fs", retry_after)
                time.sleep(retry_after + 0.1)
                continue
            if response.status_code >= 500:
                logger.warning("Server error %s for %s (attempt %s)", response.status_code, url, attempt)
                time.sleep(min(2 ** attempt, 10))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Request failed ({response.status_code}) for {url}: {response.text}")
            return response.json()
        raise RuntimeError(f"Giving up on {url} after {retries} attempts")


def iter_gold_entries(api: RiotAPI, max_pages: int) -> Iterable[Dict]:
    for division in DIVISIONS:
        for page in range(1, max_pages + 1):
            url = f"{LEAGUE_ROUTE}/lol/league/v4/entries/{QUEUE}/{TIER}/{division}"
            entries: List[Dict] = api.get_json(url, params={"page": page})
            if not entries:
                logger.debug("No more entries for %s page %s", division, page)
                break
            for entry in entries:
                yield entry


def fetch_puuid(api: RiotAPI, summoner_id: str, cache: Dict[str, str]) -> Optional[str]:
    if summoner_id in cache:
        return cache[summoner_id]
    url = f"{LEAGUE_ROUTE}/lol/summoner/v4/summoners/{summoner_id}"
    data = api.get_json(url)
    puuid = data.get("puuid")
    if puuid:
        cache[summoner_id] = puuid
    return puuid


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_match_bundle(
    api: RiotAPI,
    match_id: str,
    raw_dir: Path,
    min_duration: int,
    allowed_queue_id: Optional[int],
) -> Tuple[bool, str]:
    match_path = raw_dir / f"{match_id}.json"
    timeline_path = raw_dir / f"{match_id}_timeline.json"

    if match_path.exists() and timeline_path.exists():
        return True, "cached"

    match_json = api.get_json(f"{MATCH_ROUTE}/lol/match/v5/matches/{match_id}")
    info = match_json.get("info", {})
    queue_id = info.get("queueId")
    if allowed_queue_id is not None and queue_id != allowed_queue_id:
        logger.debug("Skipping %s (queue %s)", match_id, queue_id)
        return False, "queue"

    game_duration = int(info.get("gameDuration", 0))
    if game_duration < min_duration:
        logger.debug("Skipping %s (duration %ss)", match_id, game_duration)
        return False, "duration"

    ensure_directory(raw_dir)

    match_path.write_text(json.dumps(match_json, indent=2, sort_keys=True), encoding="utf-8")

    if timeline_path.exists():
        return True, "stored"

    try:
        timeline_json = api.get_json(f"{MATCH_ROUTE}/lol/match/v5/matches/{match_id}/timeline")
    except RuntimeError as exc:
        logger.warning("Timeline unavailable for %s: %s", match_id, exc)
        return False, "timeline"

    timeline_path.write_text(json.dumps(timeline_json, indent=2, sort_keys=True), encoding="utf-8")
    return True, "stored"


def load_existing_matchlist(path: Path) -> Tuple[List[str], set[str]]:
    if not path.is_file():
        return [], set()
    data = json.loads(path.read_text(encoding="utf-8"))
    ordered_ids = list(data)
    return ordered_ids, set(ordered_ids)


def write_matchlist(path: Path, match_ids: Sequence[str]) -> None:
    path.write_text(json.dumps(match_ids, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-match-count", type=int, default=DEFAULT_TARGET_MATCHES, help="Total unique matches to keep on disk.")
    parser.add_argument("--matches-per-player", type=int, default=DEFAULT_MATCHES_PER_PLAYER, help="How many matches to request per player (ids endpoint, capped at 100).")
    parser.add_argument("--history-window", type=int, default=200, help="Maximum depth of match history to scan per player (start offsets).")
    parser.add_argument("--max-pages-per-division", type=int, default=10, help="Pages to walk per division (25 players per page).")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory where match JSON and timelines are stored.")
    parser.add_argument("--matchlist", default="matchlist.json", help="Path to matchlist file to update.")
    parser.add_argument("--min-duration", type=int, default=DEFAULT_MIN_DURATION, help="Minimum game duration (seconds) to keep.")
    parser.add_argument("--queue-id", type=int, default=DEFAULT_QUEUE_ID, help="Queue ID to keep (420 ranked solo/duo). Use -1 to keep all queues.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

    api_key = load_api_key()
    if not api_key:
        logger.error("RIOT_API_KEY not found. Set it in the environment or .env file.")
        return 1

    api = RiotAPI(api_key)
    raw_dir = Path(args.raw_dir)
    matchlist_path = Path(args.matchlist)

    ordered_ids, known_ids = load_existing_matchlist(matchlist_path)
    logger.info("Starting with %d existing matches", len(ordered_ids))

    puuid_cache: Dict[str, str] = {}
    new_matches: List[str] = []
    skip_stats: Counter[str] = Counter()

    batch_size = max(1, min(args.matches_per_player, 100))
    if batch_size < args.matches_per_player:
        logger.warning("matches-per-player capped at %d due to Riot API limits", batch_size)

    history_window = max(batch_size, args.history_window)
    queue_filter = None if args.queue_id < 0 else args.queue_id

    for entry in iter_gold_entries(api, args.max_pages_per_division):
        puuid = entry.get("puuid")
        if puuid:
            logger.debug("Using ladder puuid for %s", entry.get("summonerName"))
        else:
            summoner_id = entry.get("summonerId") or entry.get("encryptedSummonerId")
            if not summoner_id:
                logger.debug("Entry missing summonerId/puuid: %s", entry)
                continue
            logger.debug("Fetching puuid for %s (%s)", entry.get("summonerName"), summoner_id)
            puuid = fetch_puuid(api, summoner_id, puuid_cache)
        if not puuid:
            logger.debug("Missing puuid for ladder entry: %s", entry)
            continue
        for start in range(0, history_window, batch_size):
            try:
                match_ids: List[str] = api.get_json(
                    f"{MATCH_ROUTE}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                    params={"start": start, "count": batch_size},
                )
            except RuntimeError as exc:
                logger.warning("Unable to fetch match ids for %s (start=%d): %s", puuid, start, exc)
                skip_stats["api_error"] += 1
                break

            if not match_ids:
                break

            for match_id in match_ids:
                if match_id in known_ids:
                    skip_stats["duplicate_id"] += 1
                    continue
                try:
                    stored, reason = download_match_bundle(api, match_id, raw_dir, args.min_duration, queue_filter)
                except RuntimeError as exc:
                    logger.warning("Failed to store %s: %s", match_id, exc)
                    skip_stats["download_error"] += 1
                    continue
                if not stored:
                    skip_stats[f"skip_{reason}"] += 1
                    continue
                known_ids.add(match_id)
                ordered_ids.append(match_id)
                new_matches.append(match_id)
                logger.info("Stored %s (%d/%d)", match_id, len(known_ids), args.target_match_count)
                if len(known_ids) >= args.target_match_count:
                    break
            if len(known_ids) >= args.target_match_count:
                break
        if len(known_ids) >= args.target_match_count:
            break

    if new_matches:
        write_matchlist(matchlist_path, ordered_ids)
        logger.info("Updated matchlist with %d new matches (total %d).", len(new_matches), len(ordered_ids))
    else:
        logger.info("No new matches downloaded.")

    if skip_stats:
        logger.info("Skip summary: %s", dict(skip_stats))

    if len(known_ids) < args.target_match_count:
        logger.warning(
            "Only gathered %d matches (target %d). Increase max pages or matches per player and rerun.",
            len(known_ids),
            args.target_match_count,
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Polymarket Gamma API + CLOB API fetchers.
Parses temperature bucket markets and gets live CLOB mid-prices.
"""
import re
import json
import logging
import requests
from datetime import datetime, date
from config import GAMMA_API, CLOB_API, CITY_ALIASES, MIN_MARKET_VOLUME_USDC

logger = logging.getLogger(__name__)

# ── Market question parser ────────────────────────────────────────────────────

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_question(question: str) -> dict | None:
    """
    Parse a Polymarket temperature market question.

    Handles both daily and weekly/range markets:
      Daily:  "Will NYC reach 80°F on March 22?"
      Weekly: "Will NYC reach 80°F between March 18-24?"
              "Highest temp in NYC this week above 75°F?"

    Returns dict with:
      city, target_date, bucket_lo, bucket_hi, bucket_unit,
      market_type ('daily' | 'weekly'),
      target_date_end (only for weekly — last day of range)
    or None if the question can't be parsed.
    """
    q = question.lower()

    # Find city
    city = None
    for alias, canonical in sorted(CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            city = canonical
            break
    if not city:
        return None

    # Determine unit
    unit = "F" if ("°f" in q or re.search(r'\d+°[fF]', question)) else "C"
    # For US cities, default to F even if no explicit unit (market implies it)
    # (handled by caller using city config)

    # ── Try weekly / date-range market first ────────────────────────────────
    # Pattern: "between March 18-24" or "March 18-24, 2026"
    range_match = re.search(
        r'(\w+)\s+(\d{1,2})[\s\-–]+(\d{1,2})(?:,?\s*(\d{4}))?', q
    )
    week_match = re.search(r'this\s+week', q)

    if range_match and not re.search(r'between\s+\d', q):
        month_str = range_match.group(1)
        day_start = int(range_match.group(2))
        day_end   = int(range_match.group(3))
        year_str  = range_match.group(4)
        month = _MONTHS.get(month_str)
        if month:
            year = int(year_str) if year_str else datetime.utcnow().year
            if not year_str:
                now = datetime.utcnow()
                if month < now.month:
                    year = now.year + 1
            try:
                target_date     = date(year, month, day_start)
                target_date_end = date(year, month, day_end)
                if target_date_end >= target_date:
                    # Parse bucket for weekly market
                    bucket = _parse_bucket(question, unit)
                    if bucket:
                        bucket.update({
                            "city": city,
                            "target_date": target_date,
                            "target_date_end": target_date_end,
                            "market_type": "weekly",
                        })
                        return bucket
            except ValueError:
                pass

    # ── Daily market ────────────────────────────────────────────────────────
    # Find date: "on March 22" or "on March 22, 2026"
    date_match = re.search(r'on\s+(\w+)\s+(\d{1,2})(?:,?\s*(\d{4}))?', q)
    if not date_match:
        return None
    month_str, day_str, year_str = (date_match.group(1),
                                    date_match.group(2),
                                    date_match.group(3))
    month = _MONTHS.get(month_str)
    if not month:
        return None
    year = int(year_str) if year_str else datetime.utcnow().year
    # Handle year boundary: if parsed month is before current month and no year given,
    # the market likely refers to next year (e.g. January market parsed in December)
    if not year_str:
        now = datetime.utcnow()
        if month < now.month:
            year = now.year + 1
    try:
        target_date = date(year, month, int(day_str))
    except ValueError:
        return None

    bucket = _parse_bucket(question, unit)
    if not bucket:
        return None

    bucket.update({
        "city": city,
        "target_date": target_date,
        "market_type": "daily",
    })
    return bucket


def _parse_bucket(question: str, default_unit: str) -> dict | None:
    """Parse the temperature bucket portion of a question string."""
    unit = default_unit

    # "between X-Y°F" or "between X-Y°C"
    m = re.search(r'between\s+(-?\d+(?:\.\d+)?)[–\-](-?\d+(?:\.\d+)?)\s*°?([FCfc])?',
                  question, re.IGNORECASE)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if m.group(3):
            unit = m.group(3).upper()
        return {"bucket_lo": lo, "bucket_hi": hi, "bucket_unit": unit}

    # "X°C or below" / "X°F or below"
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*°([FCfc])\s+or\s+below', question, re.IGNORECASE)
    if m:
        return {"bucket_lo": None, "bucket_hi": float(m.group(1)),
                "bucket_unit": m.group(2).upper()}

    # "X°C or higher" / "X°F or higher"
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*°([FCfc])\s+or\s+higher', question, re.IGNORECASE)
    if m:
        return {"bucket_lo": float(m.group(1)), "bucket_hi": None,
                "bucket_unit": m.group(2).upper()}

    # "reach X°F" / "exceed X°C"
    m = re.search(r'(?:reach|exceed)\s+(-?\d+(?:\.\d+)?)\s*°([FCfc])',
                  question, re.IGNORECASE)
    if m:
        return {"bucket_lo": float(m.group(1)), "bucket_hi": None,
                "bucket_unit": m.group(2).upper()}

    # "be X°C on" / "be X°F on" (exact degree bucket — half-degree range each side)
    m = re.search(r'be\s+(-?\d+(?:\.\d+)?)\s*°([FCfc])\s+on', question, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        return {"bucket_lo": val - 0.5, "bucket_hi": val + 0.5, "bucket_unit": unit}

    return None


def parse_clob_tokens(raw) -> list[str]:
    """Parse clobTokenIds which may arrive as a JSON string or a list."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back: extract long digit sequences
        tokens = re.findall(r'\d{30,}', raw)
        return tokens
    return []


# ── Gamma API fetcher ─────────────────────────────────────────────────────────

def fetch_temperature_markets() -> list[dict]:
    """
    Fetch all active temperature/weather markets from Polymarket.
    Returns list of parsed market dicts.
    Raises on API failure — no fake fallback.
    """
    all_markets = []
    batch = 100
    for offset in range(0, 10000, batch):
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset,
                "order": "endDate",
                "ascending": "true",
            },
            timeout=20,
        )

        if resp.status_code == 422:
            print(f"Parando busca no offset {offset}")
            break
            
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_markets.extend(data)
        logger.debug("Fetched %d markets (offset=%d)", len(all_markets), offset)

    logger.info("Total active markets fetched: %d", len(all_markets))
    print("DEBUG TEST 123456")

    # Filter to temperature/weather markets with sufficient liquidity
    temp_kw = [
     "temperature",
    "weather",
    "°f",
    "°c",
    "rain",
    "precipitation",
    "hottest",
    "coldest"
    ]
    temp_markets = []
    skipped_thin = 0
    for m in all_markets:
        if not any(kw in m.get("question", "").lower() for kw in temp_kw):
            continue
        # Spread filter first: if bid/ask spread >20%, market is untradeable regardless of volume
        try:
            bid = float(m.get("bestBid") or 0)
            ask = float(m.get("bestAsk") or 1)
            if bid > 0 and ask > 0 and (ask - bid) > 0.20:
                skipped_thin += 1
                logger.debug("Skipping wide-spread market (%.0f%% spread): %s",
                             (ask - bid) * 100, m.get("question", "")[:60])
                continue
        except (TypeError, ValueError):
            pass
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
        if vol < MIN_MARKET_VOLUME_USDC:
            skipped_thin += 1
            logger.debug("Skipping thin market (vol=$%.0f): %s", vol, m.get("question","")[:60])
            continue
        temp_markets.append(m)
    logger.info("Temperature markets found: %d (%d skipped — volume < $%.0f)",
                len(temp_markets), skipped_thin, MIN_MARKET_VOLUME_USDC)

    # Parse each market
    parsed = []

    # DEBUG: mostrar as primeiras perguntas encontradas
    for m in temp_markets[:50]:
        print("TEMP MARKET:", m.get("question"))

    # Loop normal do bot
    for m in temp_markets:
        question = m.get("question", "")
        market_id = m.get("conditionId") or m.get("id", "")

        if not market_id or not question:
            continue

        parsed_q = parse_question(question)

        if not parsed_q:
            logger.debug("Could not parse: %s", question[:80])
            continue

        tokens = parse_clob_tokens(m.get("clobTokenIds", "[]"))

        if not tokens:
            logger.debug("No CLOB tokens for: %s", question[:80])
            continue

        # outcomePrices[0] = YES price, tokens[0] = YES token
        yes_token = tokens[0]

        parsed.append({
            "market_id": market_id,
            "question": question,
            "city": parsed_q["city"],
            "target_date": parsed_q["target_date"],
            "target_date_end": parsed_q.get("target_date_end"),
            "market_type": parsed_q.get("market_type", "daily"),
            "bucket_lo": parsed_q["bucket_lo"],
            "bucket_hi": parsed_q["bucket_hi"],
            "bucket_unit": parsed_q["bucket_unit"],
            "clob_token_yes": yes_token,
            "outcome_prices": m.get("outcomePrices", []),
            "best_bid": m.get("bestBid"),
            "best_ask": m.get("bestAsk"),
            "last_trade": m.get("lastTradePrice"),
        })
    
    print("TEMP_MARKETS =", len(temp_markets))
    print("PARSED_MARKETS =", len(parsed))
    
    logger.info("Successfully parsed %d temperature markets", len(parsed))
    return parsed


# ── CLOB price fetcher ────────────────────────────────────────────────────────

def get_clob_mid(token_id: str) -> float:
    """
    Fetch live CLOB midpoint for a YES token.
    Returns float in [0, 1].
    Raises requests.HTTPError on failure.
    """
    resp = requests.get(
        f"{CLOB_API}/midpoint",
        params={"token_id": token_id},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    mid = data.get("mid")
    if mid is None:
        raise ValueError(f"No mid in CLOB response for token {token_id[:20]}")
    return float(mid)


def get_clob_orderbook(token_id: str) -> dict:
    """
    Fetch CLOB order book for a YES token.
    Returns dict with 'bids' and 'asks' lists.
    """
    resp = requests.get(
        f"{CLOB_API}/book",
        params={"token_id": token_id},
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json()


def get_market_prices(market: dict) -> dict:
    """
    Fetch live bid, ask, and mid for a YES token from the CLOB orderbook.
    Returns {mid, bid, ask} — any value may be None on failure.

    Use this instead of get_market_mid() when you need actual entry prices
    (ask for YES bets, 1-bid for NO bets) to correctly account for spread.
    """
    token = market.get("clob_token_yes", "")
    result: dict = {"mid": None, "bid": None, "ask": None}

    try:
        book = get_clob_orderbook(token)
        raw_bids = book.get("bids", [])
        raw_asks = book.get("asks", [])
        # Best bid = highest bid price; best ask = lowest ask price
        if raw_bids:
            result["bid"] = max(float(b["price"]) for b in raw_bids)
        if raw_asks:
            result["ask"] = min(float(a["price"]) for a in raw_asks)
        if result["bid"] is not None and result["ask"] is not None:
            result["mid"] = (result["bid"] + result["ask"]) / 2
        elif result["bid"] is not None:
            result["mid"] = result["bid"]
        elif result["ask"] is not None:
            result["mid"] = result["ask"]
        return result
    except Exception as e:
        logger.warning("CLOB orderbook failed for %s: %s — falling back to midpoint",
                       token[:20] if token else "?", e)

    # Fallback 1: CLOB midpoint endpoint
    try:
        result["mid"] = get_clob_mid(token)
    except Exception:
        pass

    # Fallback 2: Gamma API bid/ask
    try:
        if market.get("best_bid"):
            result["bid"] = float(market["best_bid"])
        if market.get("best_ask"):
            result["ask"] = float(market["best_ask"])
        if result["mid"] is None and result["bid"] and result["ask"]:
            result["mid"] = (result["bid"] + result["ask"]) / 2
    except (TypeError, ValueError):
        pass

    return result


def get_market_mid(market: dict) -> float | None:
    """
    Get the best available mid price for a market, trying CLOB first then
    falling back to Gamma API prices.
    Does NOT swallow CLOB errors — caller decides whether to skip the market.
    """
    # Try live CLOB midpoint
    try:
        return get_clob_mid(market["clob_token_yes"])
    except Exception as e:
        logger.warning("CLOB mid failed for %s: %s — using Gamma fallback",
                       market.get("market_id", "?")[:20], e)

    # Fall back to Gamma bid/ask mid
    try:
        bid = float(market["best_bid"]) if market.get("best_bid") else None
        ask = float(market["best_ask"]) if market.get("best_ask") else None
        if bid and ask:
            return (bid + ask) / 2
    except (TypeError, ValueError):
        pass

    # Fall back to last trade price
    try:
        return float(market["last_trade"]) if market.get("last_trade") else None
    except (TypeError, ValueError):
        return None

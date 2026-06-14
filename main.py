#!/usr/bin/env python3
"""
Polymarket Temperature Trading Bot — CLI

Commands:
  --scan                 Fetch live temperature markets, run signal pipeline, paper trade edges
  --scan-tsa             Fetch live TSA passenger markets, run TSA signal pipeline, paper trade edges
  --backfill             Pull 180 days of history, bias corrections, climatology
  --nowcast              Check live mid-day temperatures, update position confidence
  --resolve              Fetch today's final temperatures, settle open trades
  --monitor              Deprecated (stop-loss disabled)
  --stats                Full metrics dashboard
  --positions            Show all open positions
  --history              Show resolved trade history
  --cities               List configured cities and station status
  --export-calibration   Export calibration CSV (model vs market Brier scores)
  --dry-run              Use with --scan or --resolve to simulate without writing

Concurrency:
  --scan and --backfill use a lockfile (polymarket_bot.lock) to prevent
  duplicate instances. --nowcast is safe to run alongside anything.
"""
import argparse
import fcntl
import logging
import os
import sys
from datetime import date, timedelta
from config import CITIES, BACKFILL_DAYS, MIN_HISTORY_DAYS, OPENMETEO_MODELS
import db

LOCK_FILE = os.path.join(os.path.dirname(__file__), "polymarket_bot.lock")


def _acquire_scan_lock():
    """
    Acquire an exclusive lockfile to prevent two --scan instances running
    simultaneously. Returns (lock_fd, acquired). --nowcast does not use this.

    PID-aware: if the lockfile contains a PID from a dead process, it is
    stale and we clear it automatically so a new scan can proceed.
    This prevents the bot getting permanently stuck after a crash.
    """
    import signal as _signal

    # Check for a stale lockfile from a previously crashed process
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                old_pid = int(f.read().strip())
            # Check if the PID is still alive
            os.kill(old_pid, 0)   # raises OSError if process is dead
            # Process is alive — lock is legitimately held
        except (ValueError, OSError):
            # PID file empty/corrupt OR process is dead → stale lock
            logger.info("Removing stale lockfile (old PID no longer running)")
            try:
                os.unlink(LOCK_FILE)
            except OSError:
                pass

    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd, True
    except BlockingIOError:
        lock_fd.close()
        return None, False


def _release_scan_lock(lock_fd):
    if lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _run_daily_reconciliation_if_due(live: bool):
    if not live:
        return
    try:
        from ops_state import should_run_daily_reconcile, mark_daily_reconcile
        if not should_run_daily_reconcile():
            return
        from broker.live_broker import get_clob_balance, get_polymarket_positions_value_usd
        cash = float(get_clob_balance() or 0.0)
        pos_val = float(get_polymarket_positions_value_usd() or 0.0)
        remote_total = cash + pos_val
        local_total = float(db.get_bankroll()) + sum(float(t.get("size_usdc") or 0.0) for t in db.get_open_trades())
        drift = remote_total - local_total
        drift_pct = (abs(drift) / max(1.0, remote_total)) * 100.0
        msg = (f"daily reconcile remote=${remote_total:.2f} local=${local_total:.2f} "
               f"drift=${drift:+.2f} ({drift_pct:.2f}%)")
        if drift_pct >= 5.0:
            logger.error("ALERT %s", msg)
            db.log_event("RECONCILE_DRIFT", msg)
        else:
            logger.info("%s", msg)
            db.log_event("RECONCILE_OK", msg)
        mark_daily_reconcile()
    except Exception as e:
        logger.warning("Daily reconciliation failed: %s", e)


def _signal_health_policy():
    try:
        from ops_state import get_datasource_health
        ds = get_datasource_health("openmeteo")
        state = ds.get("state", "ok")
        if state == "offline":
            return {"skip": True, "mult": 0.0, "state": state}
        if state == "degraded":
            return {"skip": False, "mult": 0.5, "state": state}
    except Exception:
        pass
    return {"skip": False, "mult": 1.0, "state": "ok"}

G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
B   = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"


# ── --backfill ────────────────────────────────────────────────────────────────

def cmd_backfill():
    """
    For each city:
      1. Pull 180 days of historical ASOS daily-max temps
      2. Pull 180 days of Open-Meteo Archive actuals (ERA5, as historical model proxy)
      3. Pull 180 days of each model's forecast (uses archive as proxy)
      4. Recompute bias corrections
    """
    from data.noaa import fetch_asos_daily_max
    from data.openmeteo import fetch_historical_actuals, fetch_all_models
    from signals.bias_corrector import recompute_bias

    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=BACKFILL_DAYS)).isoformat()

    print(f"\n{B}{C}BACKFILL: {start_date} → {end_date}{RST}")
    print(f"  Cities: {len(CITIES)}  |  Models: {list(OPENMETEO_MODELS.keys())}\n")

    for city, cfg in CITIES.items():
        icao = cfg["icao"]
        asos = cfg["asos_station"]
        lat, lon = cfg["lat"], cfg["lon"]
        tz = cfg["timezone"]

        print(f"{B}{'─'*55}{RST}")
        print(f"  {B}{city}{RST} ({icao} / {asos})")

        # Ensure station is in DB
        db.upsert_station(icao, city, lat, lon, tz, cfg["uses_fahrenheit"])

        # ── Step 1: ASOS historical daily max ──
        print(f"  [1/5] ASOS historical obs...")
        try:
            asos_data = fetch_asos_daily_max(asos, start_date, end_date)
            count_asos = 0
            for obs_date, high_c in asos_data.items():
                db.upsert_historical_obs(icao, obs_date, high_c, "asos")
                count_asos += 1
            print(f"        {G}✓ {count_asos} days stored{RST}")
        except Exception as e:
            print(f"        {R}✗ ASOS failed: {e}{RST}")
            logger.error("ASOS backfill failed for %s: %s", icao, e)

        # ── Step 2: Open-Meteo Archive (ERA5 reanalysis as ground truth) ──
        print(f"  [2/5] Open-Meteo Archive (ERA5) actuals...")
        try:
            archive_data = fetch_historical_actuals(lat, lon, start_date, end_date, tz)
            count_arch = 0
            for obs_date, high_c in archive_data.items():
                db.upsert_historical_obs(icao, obs_date, high_c, "openmeteo_archive")
                count_arch += 1
            print(f"        {G}✓ {count_arch} days stored{RST}")
        except Exception as e:
            print(f"        {R}✗ Archive failed: {e}{RST}")
            logger.error("Archive backfill failed for %s: %s", icao, e)

        # ── Step 3: Live model forecasts for upcoming tradeable dates ──
        # Bias corrections accumulate as we run daily scans.
        # Try today, then tomorrow (model endpoints advance past midnight UTC).
        print(f"  [3/5] Storing live model forecasts...")
        stored_any = False
        for days_ahead in range(0, 3):
            forecast_date = (date.today() + timedelta(days=days_ahead)).isoformat()
            try:
                forecasts = fetch_all_models(lat, lon, forecast_date, tz)
                for model_name, predicted_c in forecasts.items():
                    db.insert_forecast(icao, forecast_date, model_name, predicted_c)
                summary = "  ".join(f"{m[:4]}={v:.1f}" for m, v in forecasts.items())
                print(f"        {forecast_date}  {summary}")
                stored_any = True
            except Exception as e:
                logger.debug("Forecast for %s %s: %s", icao, forecast_date, e)
        if stored_any:
            print(f"        {G}✓ forecasts stored{RST}")
        else:
            print(f"        {Y}⚠ no forecasts available{RST}")
            logger.warning("All forecast dates failed for %s", icao)

        # ── Step 4: Recompute bias corrections ──
        print(f"  [4/5] Computing bias corrections...")
        try:
            biases = recompute_bias(icao)
            n_biases = sum(len(months) for months in biases.values())
            if n_biases > 0:
                print(f"        {G}✓ {n_biases} bias entries computed{RST}")
            else:
                n_days = db.count_historical_obs(icao)
                if n_days < MIN_HISTORY_DAYS:
                    print(f"        {Y}⚠ Only {n_days} days — need {MIN_HISTORY_DAYS} "
                          f"before trading (warming_up){RST}")
                else:
                    print(f"        {Y}⚠ Bias computed but no model/actual overlap yet{RST}")
        except Exception as e:
            print(f"        {R}✗ Bias computation failed: {e}{RST}")
            logger.error("Bias failed for %s: %s", icao, e)

        # ── Step 5: Climatological baseline ──
        print(f"  [5/5] Fetching 30-year climatological baseline...")
        try:
            from data.climatology import fetch_climatology
            climo = fetch_climatology(lat, lon, tz)
            for month, stats in climo.items():
                db.upsert_climatology(
                    icao, month, stats["mean_c"], stats["std_c"],
                    stats["p10_c"], stats["p90_c"], stats["sample_years"],
                )
            print(f"        {G}✓ {len(climo)} months of climatology stored{RST}")
        except Exception as e:
            print(f"        {R}✗ Climatology failed: {e}{RST}")
            logger.error("Climatology failed for %s: %s", icao, e)

        n_obs = db.count_historical_obs(icao)
        st = db.get_station(icao)
        status = st["status"] if st else "?"
        status_c = G if status == "ready" else Y
        print(f"  → {status_c}{status}{RST}  ({n_obs} obs days)\n")

    pruned = db.prune_old_forecasts(days_to_keep=90)
    if pruned:
        print(f"  {DIM}Pruned {pruned} old forecast rows (>90 days){RST}")
    print(f"{B}Backfill complete.{RST}\n")


# ── --scan ────────────────────────────────────────────────────────────────────

def cmd_scan(dry_run=False, live=False, opportunistic=False):
    """
    1. Fetch all live temperature markets from Polymarket
    2. For each market, get live CLOB price
    3. Fetch 5-model forecasts from Open-Meteo
    4. Apply bias corrections
    5. Run edge calculator
    6. Paper trade (or live trade if --live) any edges found
    """
    from data.polymarket import fetch_temperature_markets, get_market_prices
    from data.openmeteo import fetch_all_models
    from signals.bias_corrector import get_corrected_ensemble, station_is_ready, get_model_weights
    from signals.edge_calculator import compute_edge
    from signals.confidence_tier import apply_tier_to_signal
    from signals.neighbor_check import get_neighbor_penalty, clear_session_cache
    clear_session_cache()  # fresh cache for each scan run
    from broker.paper_broker import execute_paper_trade
    from broker.correlation_filter import correlation_allows_trade
    if live:
        from broker.live_broker import execute_live_trade

    print(f"\n{B}{C}{'='*58}{RST}")
    scan_label = "SCAN-OPPORTUNISTIC" if opportunistic else "SCAN"
    print(f"{B}{C}  {scan_label} — {date.today().isoformat()}{RST}")
    print(f"{B}{C}{'='*58}{RST}")
    if dry_run:
        print(f"  {Y}DRY RUN — no trades will be written{RST}")

    _run_daily_reconciliation_if_due(live=live)

    # Opportunistic mode guardrails: only run if there's meaningful free capital,
    # book isn't already overly crowded, and we're outside cooldown.
    if opportunistic:
        from config import (
            OPPORTUNISTIC_MIN_FREE_BANKROLL_USDC,
            OPPORTUNISTIC_COOLDOWN_MINUTES,
            OPPORTUNISTIC_MAX_OPEN_TRADES,
        )
        now_ts = __import__("time").time()
        last_ts_raw = db.get_kv("last_opportunistic_scan_ts")
        if last_ts_raw:
            try:
                elapsed = now_ts - float(last_ts_raw)
            except (TypeError, ValueError):
                elapsed = 10**9
            if elapsed < OPPORTUNISTIC_COOLDOWN_MINUTES * 60:
                print(f"  {DIM}Skipping opportunistic scan: cooldown "
                      f"({elapsed/60:.1f}m < {OPPORTUNISTIC_COOLDOWN_MINUTES}m){RST}\n")
                return

        bankroll_now = db.get_bankroll()
        if bankroll_now < OPPORTUNISTIC_MIN_FREE_BANKROLL_USDC:
            print(f"  {DIM}Skipping opportunistic scan: free bankroll ${bankroll_now:.2f} "
                  f"< ${OPPORTUNISTIC_MIN_FREE_BANKROLL_USDC:.2f}{RST}\n")
            return

        open_count = len(db.get_open_trades())
        if open_count >= OPPORTUNISTIC_MAX_OPEN_TRADES:
            print(f"  {DIM}Skipping opportunistic scan: open trades {open_count} "
                  f">= cap {OPPORTUNISTIC_MAX_OPEN_TRADES}{RST}\n")
            return

    # Live mode: sync bankroll from actual CLOB balance so sizing stays accurate
    if live and not dry_run:
        try:
            from broker.live_broker import get_clob_balance
            clob_bal = get_clob_balance()
            if clob_bal > 0:
                db.set_bankroll(clob_bal)
                print(f"  {C}CLOB balance synced: ${clob_bal:.2f} USDC{RST}")
            else:
                print(f"  {Y}⚠ CLOB balance returned $0 — keeping DB bankroll{RST}")
        except Exception as _e:
            logger.warning("CLOB balance sync failed: %s — keeping DB bankroll", _e)

    # Auto-resolve expired trades before scanning for new ones
    # This frees correlation cap slots and updates bankroll before sizing new bets
    from broker.position_manager import resolve_expired_trades
    resolved = resolve_expired_trades(dry_run=dry_run)
    if resolved:
        won  = sum(1 for r in resolved if r.get("outcome") == "won")
        lost = sum(1 for r in resolved if r.get("outcome") == "lost")
        pnl  = sum(r.get("pnl", 0) or 0 for r in resolved)
        print(f"  {C}Auto-resolved {len(resolved)} trades: "
              f"{G}{won}W{RST}/{R}{lost}L{RST}  pnl={G if pnl >= 0 else R}${pnl:+.2f}{RST}")
        # In live mode, immediately redeem and re-sync after PM resolution so
        # newly free capital is available for this same scan pass.
        if live and not dry_run:
            try:
                from broker.live_broker import redeem_positions
                redeem = redeem_positions()
                claimed = redeem.get("usdc_claimed", 0) or 0
                if claimed > 0:
                    print(f"  {G}Auto-redeemed: +${claimed:.4f} USDC{RST}")
                    try:
                        from broker.live_broker import get_clob_balance
                        clob_bal = get_clob_balance()
                        if clob_bal > 0:
                            db.set_bankroll(clob_bal)
                    except Exception as _sync_err:
                        logger.warning("Post-redeem bankroll sync failed: %s", _sync_err)
            except Exception as _redeem_err:
                logger.warning("Auto-redeem after resolve failed: %s", _redeem_err)

    print(f"  Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")

    # 1. Fetch markets
    print(f"  Fetching active markets from Polymarket...")
    try:
        markets = fetch_temperature_markets()
        print("MARKETS RECEIVED =", len(markets))
    except Exception as e:
        logger.error("Market fetch FAILED: %s", e)
        print(f"  {R}Market fetch failed: {e}{RST}")
        sys.exit(1)
    print(f"  {G}✓ {len(markets)} temperature markets{RST}\n")
    if opportunistic:
        from config import OPPORTUNISTIC_MIN_MARKETS
        if len(markets) < OPPORTUNISTIC_MIN_MARKETS:
            print(f"  {DIM}Skipping opportunistic trading pass: only {len(markets)} markets "
                  f"(min {OPPORTUNISTIC_MIN_MARKETS}){RST}\n")
            return
        db.set_kv("last_opportunistic_scan_ts", str(__import__("time").time()))

    # Group by (city, target_date) for ensemble building
    from collections import defaultdict
    import re
    grouped = defaultdict(list)
    for m in markets:
        city = m.get("city", "")
        td   = str(m.get("target_date", ""))
        if city and td:
            grouped[(city, td)].append(m)
          
# Escolha manual da cidade
available_cities = sorted(set(city for (city, _) in grouped.keys()))

print("\nCidades disponíveis:")
for i, city_name in enumerate(available_cities, 1):
    print(f"{i}. {city_name}")

choice = input("\nEscolha a cidade (número): ").strip()

try:
    selected_city = available_cities[int(choice) - 1]
except:
    print("Cidade inválida.")
    return

print(f"\nCidade selecionada: {selected_city}\n")          

    trades_placed = 0
    traded_ids = set()

    today_str    = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

  for (city, target_date), bucket_markets in sorted(grouped.items()):

    if city != selected_city:
        continue

    if city not in CITIES:
        continue
        # Skip anything that is not TODAY
        if target_date != today_str:
            logger.debug("Skipping date %s %s (limit: TODAY only)", city, target_date)
            continue
        # Same-day markets: always attempt — nowcast weight naturally goes from 0→1
        # through the day and influences sizing in the edge calculator, but we don't
        # hard-gate on it so morning scans can still enter positions.
        cfg  = CITIES[city]
        icao = cfg["icao"]

        # Ensure station is registered
        db.upsert_station(icao, city, cfg["lat"], cfg["lon"],
                          cfg["timezone"], cfg["uses_fahrenheit"])

        # Check station readiness
        if not station_is_ready(icao):
            n = db.count_historical_obs(icao)
            print(f"  {Y}⚠ {city} ({icao}): warming_up ({n}/{MIN_HISTORY_DAYS} days) — "
                  f"observing only{RST}")

        # Health policy from data-source telemetry.
        _health = _signal_health_policy()
        if _health["skip"]:
            print(f"  {Y}⚠ {city}: data source offline — skipping entries for safety{RST}")
            continue

        # Fetch ensemble forecasts (shared across all buckets for this city/date)
        try:
            raw_forecasts = fetch_all_models(cfg["lat"], cfg["lon"], target_date, cfg["timezone"])
            # Store forecasts in DB for future bias computation
            for model_name, pred in raw_forecasts.items():
                db.insert_forecast(icao, target_date, model_name, pred["temp"])
        except Exception as e:
            logger.warning("Live fetch failed for %s %s: %s — trying cached forecasts",
                           city, target_date, e)
            db.log_event("FORECAST_FAILED", str(e), city=city, icao=icao)
            # Fall back to most recent cached forecasts for this city/date
            cached = db.get_forecasts_for_date(icao, target_date)
            if len(cached) >= 3:
                raw_forecasts = {r["model_name"]: {"temp": r["predicted_high_c"], "precip": 0}
                                     for r in cached}
                print(f"  {Y}⚠ {city}: using cached forecasts ({len(raw_forecasts)} models){RST}")
            else:
                print(f"  {R}Model fetch failed for {city} {target_date} "
                      f"(no cache): {e}{RST}")
                continue

        # Apply bias corrections
        # raw_forecasts is now {model: {'temp': float, 'precip': int}}
        raw_temps  = {m: v["temp"] for m, v in raw_forecasts.items()}
        raw_precip = {m: v["precip"] for m, v in raw_forecasts.items()}
        
        corrected = get_corrected_ensemble(icao, raw_temps, target_date)

        model_strs = "  ".join(f"{m[:4]}={v:.1f}" for m, v in corrected.items())
        # Add precip info to the header
        avg_precip = sum(raw_precip.values()) / len(raw_precip) if raw_precip else 0
        precip_str = f" {B}{B}RAIN:{avg_precip:.0f}%{RST}" if avg_precip > 20 else ""
        
        print(f"\n  {B}{city}{RST} {target_date}  {DIM}[{model_strs}]{RST}{precip_str}")

        # Data-driven model weights (falls back to hardcoded if insufficient history)
        _model_weights = get_model_weights(icao)

        # Short-circuit: skip all buckets if ensemble isn't in tradeable sweet_spot
        # (avoids CLOB + DB calls for every bucket when we know we won't trade)
        from signals.ensemble import compute_ensemble_stats
        try:
            _ens_check = compute_ensemble_stats(corrected, override_weights=_model_weights,
                                               target_date=target_date)
        except ValueError:
            print(f"  {DIM}  → skipping {city}: ensemble stats failed{RST}")
            continue
        if not _ens_check["tradeable"]:
            print(f"  {DIM}  → skipping {city}: ensemble {_ens_check['score']} "
                  f"(std={_ens_check['std_c']:.2f}°C — not sweet_spot){RST}")
            continue

        # Neighbor validation: check for model grid artifacts once per (city, date).
        # A single GFS fetch at a nearby reference coordinate — cached in-session.
        _neighbor_mult, _neighbor_reason = get_neighbor_penalty(
            city, _ens_check["mean_c"], target_date, cfg["timezone"]
        )
        if _neighbor_mult < 1.0:
            print(f"  {Y}⚠ {city}: {_neighbor_reason}{RST}")

        # Pre-fetch nowcast once per city (not once per bucket)
        from signals.nowcaster import nowcast_confidence, get_running_max_c
        _nw = nowcast_confidence(cfg["timezone"])
        if _nw > 0.05:
            _running_max, _temp_rate = get_running_max_c(city)
        else:
            _running_max, _temp_rate = None, None

        # Fetch open trades once per city/date group — shared across all bucket evaluations
        # to avoid N×M DB queries (N cities × M buckets × 3 callers each).
        _open_trades = db.get_open_trades()

        # For weekly markets: pre-fetch per-day corrected forecasts so weekly_market_prob
        # can use the actual ensemble mean per calendar day instead of a single mean.
        _per_day_corrected: list[dict[str, float]] | None = None
        is_weekly = any(m.get("market_type") == "weekly" for m in bucket_markets)
        if is_weekly:
            try:
                d_start = date.fromisoformat(target_date)
                d_end_str = next(
                    (str(m.get("target_date_end", "")) for m in bucket_markets
                     if m.get("target_date_end")),
                    None,
                )
                d_end = date.fromisoformat(d_end_str) if d_end_str else d_start + timedelta(days=6)
                _per_day_corrected = []
                for offset in range((d_end - d_start).days + 1):
                    day_str = (d_start + timedelta(days=offset)).isoformat()
                    try:
                        day_raw = fetch_all_models(cfg["lat"], cfg["lon"], day_str, cfg["timezone"])
                        day_corr = get_corrected_ensemble(icao, day_raw, day_str)
                    except Exception:
                        day_corr = corrected  # fall back to start-date ensemble
                    _per_day_corrected.append(day_corr)
            except Exception as e:
                logger.debug("Per-day weekly forecast fetch failed for %s: %s", city, e)
                _per_day_corrected = None

        # Collect live prices for consistency checking (populated during bucket loop)
        _scan_prices: dict[str, float] = {}

        # Process each bucket
        for market in bucket_markets:
            market["_cached_running_max_c"]     = _running_max
            market["_cached_temp_rate_c_per_h"] = _temp_rate
            market["_nowcast_fetched"] = True
            question = market.get("question", "")
            market_id = market.get("market_id", "")

            if market_id in traded_ids:
                continue

            # Get live CLOB prices (mid, bid, ask)
            try:
                prices = get_market_prices(market)
            except Exception as e:
                logger.warning("CLOB price failed for %s: %s", market_id[:16], e)
                db.log_event("CLOB_FAILED", str(e), city=city, icao=icao)
                continue

            mid = prices["mid"]
            bid = prices["bid"]
            ask = prices["ask"]

            if mid is None:
                continue
            if mid <= 0.002 or mid >= 0.998:
                continue  # Already resolved or no liquidity
            _scan_prices[market_id] = mid

            # Store/update market in DB
            db.upsert_market(
                market_id=market_id,
                city=city,
                icao=icao,
                target_date=target_date,
                question=question,
                bucket_lo=market.get("bucket_lo"),
                bucket_hi=market.get("bucket_hi"),
                bucket_unit=market.get("bucket_unit", "C"),
                clob_token_yes=market.get("clob_token_yes", ""),
            )

            # Compute edge (pass actual bid/ask for accurate entry price in Kelly sizing)
            # For weekly markets, pass per-day corrected forecasts for more accurate probability
            _pdc = _per_day_corrected if market.get("market_type") == "weekly" else None
            signal = compute_edge(market, corrected, mid, city, icao=icao,
                                  model_weights=_model_weights, bid=bid, ask=ask,
                                  per_day_corrected=_pdc, precip_prob=avg_precip)
            lo = market.get("bucket_lo")
            hi = market.get("bucket_hi")
            unit = market.get("bucket_unit", "C")
            bucket_str = f"[{lo or '-∞'},{hi or '+∞'}){unit}"

            # Record price snapshot for timing logic (every scan, regardless of edge)
            db.record_price(market_id, mid,
                            model_prob=signal["model_prob"] if signal else None,
                            edge=signal["edge"] if signal else None)

            if signal is None:
                print(f"    {DIM}{bucket_str:<22} mid={mid:.3f}  no edge{RST}")
                continue

            # Apply confidence tiering to scale Kelly size
            is_ready = station_is_ready(icao)
            station_rec = db.get_station(icao)
            hist_days = station_rec["history_days"] if station_rec else 0
            signal = apply_tier_to_signal(signal, station_ready=is_ready)
            if _health["mult"] < 1.0:
                signal["size_usdc"] = round(signal["size_usdc"] * _health["mult"], 2)
                signal["health_penalty"] = _health["mult"]

            # Apply neighbor grid-artifact penalty (computed once per city/date above)
            if _neighbor_mult < 1.0:
                signal["size_usdc"]       = round(signal["size_usdc"] * _neighbor_mult, 2)
                signal["neighbor_penalty"] = _neighbor_mult

            tier_label = f"T{signal['confidence_tier']}"
            edge_color = G if abs(signal["edge"]) >= 0.08 else Y
            climo_str = (f"  climo_dev={signal['climo_deviation_c']:+.1f}°C"
                         if signal.get("climo_deviation_c") is not None else "")
            neighbor_str = f"  {Y}⚠ngbr×{_neighbor_mult:.1f}{RST}" if _neighbor_mult < 1.0 else ""
            hp = f"  {Y}⚠health×{_health['mult']:.1f}{RST}" if _health["mult"] < 1.0 else ""
            print(f"    {bucket_str:<22} "
                  f"mid={mid:.3f}  model={signal['model_prob']:.3f}  "
                  f"edge={edge_color}{signal['edge']:+.3f}{RST}  "
                  f"std={signal['ensemble_std_c']:.2f}°C  "
                  f"${signal['size_usdc']:.2f}  {signal['direction']}  "
                  f"{DIM}{tier_label}{climo_str}{RST}{neighbor_str}{hp}")

            if signal["size_usdc"] < 1.0:
                logger.debug("Size $%.2f after tiering < $1 — skip", signal["size_usdc"])
                continue

            if not is_ready:
                print(f"    {Y}      → OBSERVE ONLY (warming up){RST}")
                continue

            # Correlation filter: cap exposure per weather region
            allowed, corr_reason = correlation_allows_trade(
                city, target_date,
                direction=signal["direction"],
                open_trades=_open_trades,
            )
            if not allowed:
                print(f"    {Y}      → skipped: {corr_reason}{RST}")
                continue

            # Per-city daily cap: all bets for the same city+date share the same
            # underlying temperature outcome. Cap total deployed to MAX_CITY_DATE_FRACTION
            # to prevent one bad city day from causing an outsized correlated loss.
            from config import MAX_CITY_DATE_FRACTION
            _city_date_deployed = sum(
                t["size_usdc"] for t in _open_trades
                if t["city"] == city and str(t["target_date"]) == target_date
            )
            _city_date_cap = MAX_CITY_DATE_FRACTION * db.get_bankroll()
            if _city_date_deployed + signal["size_usdc"] > _city_date_cap:
                _headroom = max(0.0, _city_date_cap - _city_date_deployed)
                if _headroom < 1.0:
                    print(f"    {Y}      → skipped: city-date cap hit "
                          f"(deployed=${_city_date_deployed:.2f} cap=${_city_date_cap:.2f}){RST}")
                    continue
                # Partially allowed: trim size to remaining headroom
                signal["size_usdc"] = round(_headroom, 2)
                logger.info("City-date cap: trimming size to $%.2f headroom "
                            "(deployed=%.2f cap=%.2f)", _headroom, _city_date_deployed, _city_date_cap)

            # Execute paper trade first (risk checks, bankroll, DB)
            result = execute_paper_trade(market, signal, dry_run=dry_run,
                                         open_trades=_open_trades)
            if "skipped" in result:
                print(f"    {Y}      → skipped: {result['skipped']}{RST}")
            elif result.get("dry_run"):
                print(f"    {C}      → [DRY] would trade{RST}")
            else:
                print(f"    {G}      → PAPER {result['direction']} "
                      f"${result['size_usdc']:.2f} @ {result['entry_price']:.4f} "
                      f"| roll=${result['bankroll_after']:.2f}{RST}")
                trades_placed += 1
                traded_ids.add(market_id)

                # If --live, also submit the real order to the CLOB
                if live:
                    live_result = execute_live_trade(market, signal, dry_run=dry_run)
                    if "skipped" in live_result:
                        print(f"    {R}      → LIVE ORDER skipped: {live_result['skipped']}{RST}")
                    elif live_result.get("dry_run"):
                        print(f"    {C}      → [DRY LIVE] order would submit{RST}")
                    else:
                        print(f"    {G}      → LIVE ORDER {live_result.get('order_id','?')[:12]} "
                              f"status={live_result.get('status','?')}{RST}")

        # Cross-market consistency check: find bucket arithmetic violations
        if len(_scan_prices) >= 2:
            from signals.consistency_checker import find_consistency_signals
            arb_signals = find_consistency_signals(
                bucket_markets, _scan_prices,
                unit=cfg.get("bucket_unit"),
            )
            for arb in arb_signals:
                print(f"  {Y}⚡ CONSISTENCY ARB [{arb['type']}]{RST} "
                      f"{arb['direction']} {arb['market_id'][:16]}  "
                      f"market={arb['market_prob']:.3f}  fair≈{arb['implied_fair']:.3f}  "
                      f"edge={arb['implied_edge']:+.3f}")
                logger.info("Consistency signal: %s", arb["message"])

    print(f"\n{B}Scan complete.{RST}  Trades placed: {G}{trades_placed}{RST}  "
          f"| Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")


# ── --scan-tsa ────────────────────────────────────────────────────────────────

def cmd_scan_tsa(dry_run=False):
    """
    Scan for TSA passenger market edges and paper-trade any found.

    Pipeline:
      1. Fetch active TSA markets from Polymarket
      2. Get live CLOB prices
      3. Fetch TSA historical data from tsa.gov
      4. Compute day-of-week baseline + YoY + holiday + hub weather signal
      5. Compute edge and Kelly size
      6. Paper trade if edge > MIN_EDGE
    """
    from data.polymarket_tsa import fetch_tsa_markets, get_tsa_market_prices
    from data.polymarket import get_market_prices
    from data.tsa import fetch_tsa_data
    from signals.tsa_edge_calculator import compute_tsa_edge
    from broker.paper_broker import execute_paper_trade

    print(f"\n{B}{C}{'='*58}{RST}")
    print(f"{B}{C}  SCAN-TSA — {date.today().isoformat()}{RST}")
    print(f"{B}{C}{'='*58}{RST}")
    if dry_run:
        print(f"  {Y}DRY RUN — no trades will be written{RST}")

    print(f"  Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")

    # 1. Fetch markets
    print(f"  Fetching TSA markets from Polymarket...")
    try:
        markets = fetch_tsa_markets()
    except Exception as e:
        logger.error("TSA market fetch FAILED: %s", e)
        print(f"  {R}Market fetch failed: {e}{RST}")
        return
    print(f"  {G}✓ {len(markets)} TSA markets{RST}\n")

    if not markets:
        print(f"  {Y}No active TSA passenger markets found on Polymarket.{RST}\n")
        return

    # 2. Fetch TSA historical data once (shared across all markets)
    print(f"  Fetching TSA historical data from tsa.gov...")
    try:
        from data.tsa import fetch_tsa_data
        tsa_data = fetch_tsa_data()
        print(f"  {G}✓ {len(tsa_data)} days of TSA data loaded{RST}\n")
    except Exception as e:
        logger.error("TSA data fetch failed: %s", e)
        print(f"  {R}TSA data fetch failed: {e}{RST}")
        return

    trades_placed = 0
    traded_ids = set()
    open_trades = db.get_open_trades()
    today_str    = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

    # Group by target_date so we display them together
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for m in markets:
        grouped[str(m.get("target_date", ""))].append(m)

    for target_date, date_markets in sorted(grouped.items()):
        # Skip anything that is not TODAY
        if target_date != today_str:
            logger.debug("Skipping date %s (limit: TODAY only)", target_date)
            continue

        print(f"\n  {B}TSA {target_date}{RST}")

        for market in date_markets:
            market_id = market.get("market_id", "")
            if market_id in traded_ids:
                continue

            # Get live CLOB price
            try:
                prices = get_market_prices(market)
            except Exception as e:
                logger.warning("CLOB price failed for TSA %s: %s", market_id[:16], e)
                continue

            mid = prices.get("mid")
            bid = prices.get("bid")
            ask = prices.get("ask")

            if mid is None or mid <= 0.002 or mid >= 0.998:
                continue

            # Register market in DB (reuse markets table, bucket_unit='M')
            db.upsert_market(
                market_id=market_id,
                city="TSA",
                icao="TSA",
                target_date=target_date,
                question=market.get("question", ""),
                bucket_lo=market.get("bucket_lo"),
                bucket_hi=market.get("bucket_hi"),
                bucket_unit="M",
                clob_token_yes=market.get("clob_token_yes", ""),
            )

            # Compute edge
            signal = compute_tsa_edge(market, mid, tsa_data, bid=bid, ask=ask)

            lo  = market.get("bucket_lo")
            hi  = market.get("bucket_hi")
            bucket_str = f"[{lo if lo is not None else '-∞'},{hi if hi is not None else '+∞'}M)"

            db.record_price(market_id, mid,
                            model_prob=signal["model_prob"] if signal else None,
                            edge=signal["edge"] if signal else None)

            if signal is None:
                print(f"    {DIM}{bucket_str:<22} mid={mid:.3f}  no edge{RST}")
                continue

            # Log TSA signal for calibration tracking (separate from temperature calibration)
            import uuid as _uuid
            db.record_tsa_prediction(
                pred_id=str(_uuid.uuid4()),
                market_id=market_id,
                model_prob=signal["model_prob"],
                market_prob=signal["market_prob"],
                tsa_mean_m=signal.get("tsa_mean_m"),
                tsa_std_m=signal.get("tsa_std_m"),
                hub_weather_flag=signal.get("hub_weather_flag"),
            )

            hub_flag_str = f"  {Y}⛈ hubs:{signal['hub_bad_list']}{RST}" if signal["hub_weather_flag"] else ""
            holiday_str  = f"  🗓 {signal['tsa_holiday_name']}" if signal.get("tsa_holiday_name") else ""
            print(f"    {bucket_str:<22} "
                  f"mid={mid:.3f}  model={signal['model_prob']:.3f}  "
                  f"edge={G if abs(signal['edge']) >= 0.08 else Y}{signal['edge']:+.3f}{RST}  "
                  f"${signal['size_usdc']:.2f}  {signal['direction']}"
                  f"{hub_flag_str}{holiday_str}")

            if signal["size_usdc"] < 1.0:
                continue

            # Execute paper trade (TSA uses same broker)
            market["market_type"] = "tsa"
            result = execute_paper_trade(market, signal, dry_run=dry_run,
                                         open_trades=open_trades)
            if "skipped" in result:
                print(f"    {Y}      → skipped: {result['skipped']}{RST}")
            elif result.get("dry_run"):
                print(f"    {C}      → [DRY] would trade{RST}")
            else:
                hub_log = f" hub_wx={signal['hub_weather_flag']}" if signal["hub_weather_flag"] else ""
                print(f"    {G}      → TRADED {result['direction']} "
                      f"${result['size_usdc']:.2f} @ {result['entry_price']:.4f}"
                      f"{hub_log} | roll=${result['bankroll_after']:.2f}{RST}")
                trades_placed += 1
                traded_ids.add(market_id)

    print(f"\n{B}TSA scan complete.{RST}  Trades placed: {G}{trades_placed}{RST}  "
          f"| Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")


# ── --scan-crypto ─────────────────────────────────────────────────────────────

def cmd_scan_crypto(dry_run=False):
    """
    Scan for crypto Up/Down market edges and paper-trade any found.

    Pipeline:
      1. Fetch active BTC/ETH Up/Down hourly markets from Polymarket
      2. Get live CLOB prices
      3. Fetch Deribit index price + ATM IV
      4. Use stored reference price (from first scan of this market) or
         current spot as reference if market is newly opened
      5. Compute N(d2) edge and Kelly size
      6. Paper trade if edge > MIN_EDGE
    """
    from data.polymarket_crypto import fetch_crypto_markets, get_crypto_market_prices
    from data.deribit import get_crypto_signal_inputs
    from signals.crypto_edge_calculator import compute_crypto_edge
    from broker.paper_broker import execute_paper_trade
    from datetime import datetime, timezone

    print(f"\n{B}{C}{'='*58}{RST}")
    print(f"{B}{C}  SCAN-CRYPTO — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}{RST}")
    print(f"{B}{C}{'='*58}{RST}")
    if dry_run:
        print(f"  {Y}DRY RUN — no trades will be written{RST}")
    print(f"  Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")

    # 1. Fetch markets
    print("  Fetching crypto Up/Down markets from Polymarket...")
    try:
        markets = fetch_crypto_markets()
    except Exception as e:
        logger.error("Crypto market fetch FAILED: %s", e)
        print(f"  {R}Market fetch failed: {e}{RST}")
        return

    if not markets:
        print(f"  {Y}No active crypto Up/Down markets found — markets may not be open yet.{RST}\n")
        return
    print(f"  {G}✓ {len(markets)} crypto markets{RST}\n")

    # 2. Fetch Deribit signal inputs once per asset
    print("  Fetching Deribit prices and IV...")
    deribit_inputs: dict[str, dict] = {}
    for asset in set(m["asset"] for m in markets):
        try:
            inp = get_crypto_signal_inputs(asset)
            if inp:
                deribit_inputs[asset] = inp
                print(f"  {G}✓ {asset}: spot=${inp['spot']:,.2f}  IV={inp['iv_annual']:.1%}{RST}")
            else:
                print(f"  {R}✗ {asset}: Deribit fetch failed{RST}")
        except Exception as e:
            logger.error("Deribit fetch failed for %s: %s", asset, e)
            print(f"  {R}✗ {asset}: {e}{RST}")
    print()

    if not deribit_inputs:
        print(f"  {R}No Deribit data — cannot compute signal. Exiting.{RST}\n")
        return

    trades_placed = 0
    open_trades = db.get_open_trades()

    for market in sorted(markets, key=lambda m: m["end_time"]):
        asset = market["asset"]
        market_id = market["market_id"]

        if asset not in deribit_inputs:
            continue

        inp = deribit_inputs[asset]
        spot = inp["spot"]

        # 3. Get live CLOB price
        try:
            prices = get_crypto_market_prices(market)
        except Exception as e:
            logger.warning("CLOB price failed for %s: %s", market_id[:16], e)
            continue

        mid = prices.get("mid")
        bid = prices.get("bid")
        ask = prices.get("ask")

        if mid is None or mid <= 0.005 or mid >= 0.995:
            continue

        # 4. Reference price: stored on first scan; fall back to current spot
        ref_key = f"crypto_ref_{market_id}"
        reference_price = db.get_kv(ref_key)
        if reference_price is None:
            reference_price = spot
            db.set_kv(ref_key, str(spot))
            logger.debug("Stored reference price %.2f for %s", spot, market_id[:16])
        else:
            reference_price = float(reference_price)

        # 5. Register in DB
        db.upsert_market(
            market_id=market_id,
            city=asset,
            icao=asset,
            target_date=market["target_date"],
            question=market["question"],
            bucket_lo=None,
            bucket_hi=None,
            bucket_unit="crypto",
            clob_token_yes=market.get("clob_token_yes", ""),
        )

        db.record_price(market_id, mid, model_prob=None, edge=None)

        # 6. Compute edge
        signal = compute_crypto_edge(
            market=market,
            market_implied_prob=mid,
            spot=spot,
            reference_price=reference_price,
            iv_annual=inp["iv_annual"],
            bid=bid,
            ask=ask,
        )

        end_short = market["end_time"][11:16] + " UTC"
        move_pct  = (spot - reference_price) / reference_price * 100

        if signal is None:
            print(f"  {DIM}{asset} {end_short}  spot={spot:,.0f}  ref={reference_price:,.0f} "
                  f"({move_pct:+.2f}%)  mid={mid:.3f}  no edge{RST}")
            continue

        print(f"  {asset} {end_short}  spot={spot:,.0f}  ref={reference_price:,.0f} "
              f"({move_pct:+.2f}%)  mid={mid:.3f}  "
              f"model={signal['model_prob']:.3f}  "
              f"edge={G if abs(signal['edge']) >= 0.08 else Y}{signal['edge']:+.3f}{RST}  "
              f"${signal['size_usdc']:.2f}  {signal['direction']}  "
              f"({signal['crypto_hours_rem']:.1f}h left)")

        if signal["size_usdc"] < 1.0:
            continue

        market["market_type"] = "crypto"
        result = execute_paper_trade(market, signal, dry_run=dry_run,
                                     open_trades=open_trades)
        if "skipped" in result:
            print(f"    {Y}→ skipped: {result['skipped']}{RST}")
        elif result.get("dry_run"):
            print(f"    {C}→ [DRY] would trade{RST}")
        else:
            print(f"    {G}→ TRADED {result['direction']} "
                  f"${result['size_usdc']:.2f} @ {result['entry_price']:.4f} "
                  f"| roll=${result['bankroll_after']:.2f}{RST}")
            trades_placed += 1

    print(f"\n{B}Crypto scan complete.{RST}  Trades placed: {G}{trades_placed}{RST}  "
          f"| Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")


# ── --nowcast ─────────────────────────────────────────────────────────────────

def cmd_nowcast():
    """Check live mid-day temperatures for all cities."""
    from signals.nowcaster import nowcast_confidence, get_running_max_c
    import pytz

    print(f"\n{B}{C}NOWCAST — {date.today().isoformat()}{RST}\n")
    rows = []
    for city, cfg in sorted(CITIES.items()):
        conf = nowcast_confidence(cfg["timezone"])
        if conf > 0:
            running_max, temp_rate = get_running_max_c(city)
            rate_str = (f"  {temp_rate:+.1f}°C/h" if temp_rate is not None else "")
            rows.append([
                city, cfg["icao"],
                f"{conf:.2f}",
                (f"{running_max:.1f}°C{rate_str}" if running_max is not None
                 else f"{Y}unavailable{RST}"),
            ])
        else:
            from datetime import datetime
            tz = __import__("pytz").timezone(cfg["timezone"])
            local_hr = datetime.now(tz).strftime("%H:%M")
            rows.append([city, cfg["icao"], f"{DIM}0.00{RST}",
                          f"{DIM}before 12pm local ({local_hr}){RST}"])

    from tabulate import tabulate
    print(tabulate(rows, headers=["City", "ICAO", "Confidence", "Running Max"],
                   tablefmt="rounded_outline"))
    print()


# ── --exit-scan ───────────────────────────────────────────────────────────────

def cmd_exit_scan(dry_run=False):
    """
    Review all open positions and exit early if any of these conditions are met:

    1. TAKE PROFIT   — current mark-to-market >= 3x entry cost
    2. EDGE REVERSAL — market has moved past model probability (edge flipped > MIN_EDGE)
    3. CLOSING SOON  — target_date is today or past and < 2 hours of liquidity left
    """
    from data.polymarket import get_market_prices
    from datetime import datetime, timezone, date as date_type
    import pytz

    TAKE_PROFIT_MULTIPLE  = 3.0   # exit if position worth 3x what we paid
    EDGE_REVERSAL_MIN     = 0.10  # exit if edge has flipped by at least this much
    CLOSE_SOON_HOURS      = 2     # exit if market resolves within this many hours

    open_trades = db.get_open_trades()
    if not open_trades:
        print(f"  {DIM}No open trades.{RST}")
        return

    print(f"\n{B}{C}EXIT SCAN — {date.today().isoformat()}{RST}  "
          f"({len(open_trades)} open positions)\n")

    exited = 0
    for trade in open_trades:
        trade_id    = trade["trade_id"]
        city        = trade["city"]
        direction   = trade["direction"]
        entry_price = trade["entry_price"]
        model_prob  = trade["model_prob"]
        target_date = trade["target_date"]   # stored as ISO string YYYY-MM-DD
        size        = trade["size_usdc"]

        # ── Fetch live price ──────────────────────────────────────────────────
        market_stub = {"market_id": trade["market_id"],
                       "clob_token_yes": trade.get("clob_token_yes", "")}
        try:
            prices = get_market_prices(market_stub)
        except Exception as e:
            logger.debug("exit-scan: price fetch failed for %s: %s", trade_id[:8], e)
            continue

        current_mid = prices.get("mid")
        if current_mid is None:
            continue

        # Current exit value per share (what we'd receive selling now)
        if direction == "YES":
            exit_val = prices["bid"] if prices["bid"] is not None else current_mid
        else:
            ask = prices["ask"]
            exit_val = (1.0 - ask) if ask is not None else (1.0 - current_mid)

        gain_multiple = exit_val / entry_price if entry_price > 0 else 0

        # ── Check exit conditions ─────────────────────────────────────────────

        reason = None

        # 1. Take profit
        if reason is None and gain_multiple >= TAKE_PROFIT_MULTIPLE:
            reason = f"TAKE-PROFIT {gain_multiple:.1f}x (entry={entry_price:.3f} now={exit_val:.3f})"

        # 2. Edge reversal — use stored model_prob vs current market mid
        if reason is None and model_prob is not None:
            if direction == "YES":
                current_edge = model_prob - current_mid
            else:
                current_edge = current_mid - model_prob   # positive = NO still has edge

            original_edge = abs(trade.get("edge") or 0)
            if current_edge < -EDGE_REVERSAL_MIN:
                reason = (f"EDGE-REVERSED model={model_prob:.3f} market={current_mid:.3f} "
                          f"edge={current_edge:+.3f}")

        # 3. Market closing soon
        if reason is None and target_date:
            try:
                td = date_type.fromisoformat(target_date)
                # Find city timezone to estimate local close time
                tz_str = CITIES.get(city, {}).get("timezone", "UTC")
                tz = pytz.timezone(tz_str)
                # Local midnight = market resolution; subtract CLOSE_SOON_HOURS
                local_midnight = tz.localize(
                    datetime(td.year, td.month, td.day, 23, 59)
                )
                utc_close = local_midnight.astimezone(pytz.utc)
                now_utc = datetime.now(timezone.utc)
                hours_left = (utc_close - now_utc).total_seconds() / 3600
                if 0 < hours_left < CLOSE_SOON_HOURS:
                    reason = f"CLOSING-SOON {hours_left:.1f}h left (exit into remaining liquidity)"
            except Exception:
                pass

        if reason is None:
            print(f"  {DIM}HOLD  {trade_id[:8]} | {city} {direction} "
                  f"entry={entry_price:.3f} now={exit_val:.3f} ({gain_multiple:.2f}x){RST}")
            continue

        # ── Execute exit ──────────────────────────────────────────────────────
        shares = size / entry_price
        pnl_est = shares * exit_val - size

        tag = G if pnl_est >= 0 else R
        print(f"  {tag}EXIT  {trade_id[:8]} | {city} {target_date} {direction} "
              f"entry={entry_price:.3f} → {exit_val:.3f}  "
              f"pnl={tag}${pnl_est:+.2f}{RST}  [{reason}]")

        if not dry_run:
            outcome = "won" if pnl_est >= 0 else "lost"
            # For live mode: actually submit SELL order to CLOB before marking resolved
            if not dry_run and db.get_mode() == "live":
                try:
                    from broker.live_broker import sell_position
                    token = trade.get("clob_token_yes", "")
                    if token:
                        sell_result = sell_position(token, shares, min_price=exit_val * 0.95)
                        if "error" in sell_result:
                            print(f"    {R}SELL order failed: {sell_result['error']}{RST}")
                except Exception as e:
                    logger.warning("Live sell failed for %s: %s", trade_id[:8], e)
            db.resolve_trade(trade_id, None, outcome, exit_val, outcome_source="exit_scan")
            db.log_event("EXIT_SCAN", reason, city=city, icao=trade.get("icao", ""))
            exited += 1
        else:
            print(f"    {Y}[DRY RUN] would exit{RST}")

    print(f"\n  Exited: {G}{exited}{RST} / {len(open_trades)} positions  "
          f"| Bankroll: {B}${db.get_bankroll():.2f}{RST}\n")


# ── --resolve ─────────────────────────────────────────────────────────────────

def cmd_resolve(dry_run=False):
    """Resolve all expired open trades."""
    from broker.position_manager import resolve_expired_trades

    print(f"\n{B}{C}RESOLVE — {date.today().isoformat()}{RST}")
    if dry_run:
        print(f"  {Y}DRY RUN{RST}")
    print()

    results = resolve_expired_trades(dry_run=dry_run)
    if not results:
        print(f"  {DIM}No expired trades.{RST}\n")
        return

    for r in results:
        if r.get("outcome") == "won":
            print(f"  {G}✓ WON{RST}   {r['city']} {r['date']} "
                  f"actual={r['actual_c']:.1f}°C  pnl={G}${r['pnl']:+.2f}{RST}")
        elif r.get("outcome") == "lost":
            print(f"  {R}✗ LOST{RST}  {r['city']} {r['date']} "
                  f"actual={r['actual_c']:.1f}°C  pnl={R}${r['pnl']:+.2f}{RST}")
        elif r.get("dry_run"):
            print(f"  {C}[DRY]{RST} {r['trade_id'][:8]} → {r['outcome']}")
        else:
            print(f"  {Y}?{RST} {r.get('trade_id','?')[:8]} — {r.get('status','?')}")
    print()

    # Auto-redeem winnings to CLOB wallet (live mode only)
    if not dry_run and db.get_mode() == "live":
        claimed = 0.0
        try:
            from broker.live_broker import redeem_positions
            redeem = redeem_positions()
            claimed = redeem.get("usdc_claimed", 0) or 0
            if claimed > 0:
                print(f"  {G}💰 Auto-redeemed: +${claimed:.4f} USDC returned to wallet{RST}")
            else:
                print(f"  {DIM}Auto-redeem: nothing to claim (or already claimed){RST}")
        except Exception as e:
            logger.warning("Auto-redeem failed: %s", e)
        # Keep bankroll authoritative after any resolve/redeem cycle.
        try:
            from broker.live_broker import get_clob_balance
            clob_bal = get_clob_balance()
            if clob_bal > 0:
                db.set_bankroll(clob_bal)
                print(f"  {C}CLOB balance synced: ${clob_bal:.2f} USDC{RST}")
        except Exception as e:
            logger.warning("Post-resolve balance sync failed: %s", e)
        # Capital recycling: if we resolved trades, immediately run a scan pass.
        if results:
            print(f"  {C}Capital recycle: running immediate scan after resolve...{RST}")
            lock_fd, acquired = _acquire_scan_lock()
            if not acquired:
                print(f"  {Y}Skip recycle scan: another scan is already running{RST}")
            else:
                try:
                    cmd_scan(dry_run=False, live=True)
                finally:
                    _release_scan_lock(lock_fd)
    print()


# ── --sync-positions ───────────────────────────────────────────────────────────

def cmd_sync_positions():
    """Pull actual CLOB positions and reconcile with our internal DB."""
    if db.get_mode() != "live":
        print(f"  {Y}sync-positions only available in live mode{RST}")
        return
    from broker.live_broker import sync_positions_to_db, redeem_positions
    print(f"\n{B}{C}SYNC POSITIONS — {date.today().isoformat()}{RST}\n")
    result = sync_positions_to_db()
    print(f"  CLOB positions: {result['clob_positions']}")
    print(f"  Matched to DB:  {result['synced']}")
    print(f"  Not on CLOB:    {result['not_filled']}  (orders didn't fill)")
    print()
    redeem = redeem_positions()
    claimed = redeem.get("usdc_claimed", 0)
    if claimed > 0:
        print(f"  {G}Redeemed: +${claimed:.4f} USDC{RST}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Temperature Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode",               choices=["paper", "live"], default="paper",
                        help="Which DB to use: paper (default) or live")
    parser.add_argument("--scan",               action="store_true")
    parser.add_argument("--opportunistic",      action="store_true",
                        help="Guarded scan mode for half-hour capital redeploy passes")
    parser.add_argument("--scan-tsa",           action="store_true",
                        help="Scan active TSA passenger markets and paper trade edges")
    parser.add_argument("--scan-crypto",        action="store_true",
                        help="Scan active crypto Up/Down hourly markets and paper trade edges")
    parser.add_argument("--backfill",           action="store_true")
    parser.add_argument("--nowcast",            action="store_true")
    parser.add_argument("--resolve",            action="store_true")
    parser.add_argument("--exit-scan",          action="store_true",
                        help="Exit open positions that hit take-profit, edge-reversal, or closing-soon")
    parser.add_argument("--sync-positions",     action="store_true",
                        help="Pull real CLOB positions and reconcile with DB; redeem any winnings")
    parser.add_argument("--monitor",            action="store_true")
    parser.add_argument("--stats",              action="store_true")
    parser.add_argument("--positions",          action="store_true")
    parser.add_argument("--history",            action="store_true")
    parser.add_argument("--cities",             action="store_true")
    parser.add_argument("--calibration",        action="store_true",
                        help="Show full calibration curve report (predicted vs actual win rate)")
    parser.add_argument("--export-calibration", metavar="FILE",
                        help="Export calibration CSV to FILE")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Run --scan or --resolve without writing trades/resolutions")
    parser.add_argument("--live",               action="store_true",
                        help="Use with --scan to submit real orders to Polymarket CLOB")
    parser.add_argument("--scrape-prices",      action="store_true",
                        help="Backfill hourly CLOB price history for all temp markets")
    args = parser.parse_args()

    # Switch DB before init — live mode uses live_trades.db
    db.set_mode(args.mode)
    db.init_db()

    # --mode live implies --live (real order submission)
    if args.mode == "live":
        args.live = True

    if args.scan_crypto:
        cmd_scan_crypto(dry_run=args.dry_run)
    elif args.scan_tsa:
        lock_fd, acquired = _acquire_scan_lock()
        if not acquired:
            print(f"{R}Another scan is already running (lockfile held). Exiting.{RST}")
            sys.exit(1)
        try:
            cmd_scan_tsa(dry_run=args.dry_run)
        finally:
            _release_scan_lock(lock_fd)
    elif args.backfill:
        lock_fd, acquired = _acquire_scan_lock()
        if not acquired:
            print(f"{R}Another scan/backfill is already running. Exiting.{RST}")
            sys.exit(1)
        try:
            cmd_backfill()
        finally:
            _release_scan_lock(lock_fd)
    elif args.scan:
        from ops_state import acquire_job_lock, release_job_lock, mark_job_start, mark_job_end
        lock_fd, acquired, owner_pid = acquire_job_lock("scan")
        if not acquired:
            msg = f"Skip --scan: lock held by pid={owner_pid or '?'}"
            print(f"{Y}{msg}{RST}")
            logger.warning(msg)
            db.log_event("JOB_SKIPPED_LOCK", msg)
            return
        started = mark_job_start("scan")
        try:
            cmd_scan(dry_run=args.dry_run, live=args.live, opportunistic=args.opportunistic)
            mark_job_end("scan", True, started)
        except Exception as e:
            mark_job_end("scan", False, started, str(e))
            raise
        finally:
            release_job_lock("scan", lock_fd)
    elif args.nowcast:
        # --nowcast is safe to run concurrently — no lockfile needed
        cmd_nowcast()
    elif args.resolve:
        from ops_state import acquire_job_lock, release_job_lock, mark_job_start, mark_job_end
        lock_fd, acquired, owner_pid = acquire_job_lock("resolve")
        if not acquired:
            msg = f"Skip --resolve: lock held by pid={owner_pid or '?'}"
            print(f"{Y}{msg}{RST}")
            logger.warning(msg)
            db.log_event("JOB_SKIPPED_LOCK", msg)
            return
        started = mark_job_start("resolve")
        try:
            cmd_resolve(dry_run=args.dry_run)
            mark_job_end("resolve", True, started)
        except Exception as e:
            mark_job_end("resolve", False, started, str(e))
            raise
        finally:
            release_job_lock("resolve", lock_fd)
    elif args.exit_scan:
        from ops_state import acquire_job_lock, release_job_lock, mark_job_start, mark_job_end
        lock_fd, acquired, owner_pid = acquire_job_lock("exit-scan")
        if not acquired:
            msg = f"Skip --exit-scan: lock held by pid={owner_pid or '?'}"
            print(f"{Y}{msg}{RST}")
            logger.warning(msg)
            db.log_event("JOB_SKIPPED_LOCK", msg)
            return
        started = mark_job_start("exit-scan")
        try:
            cmd_exit_scan(dry_run=args.dry_run)
            mark_job_end("exit-scan", True, started)
        except Exception as e:
            mark_job_end("exit-scan", False, started, str(e))
            raise
        finally:
            release_job_lock("exit-scan", lock_fd)
    elif getattr(args, 'sync_positions', False):
        cmd_sync_positions()
    elif args.monitor:
        print(f"\n{Y}--monitor is deprecated: stop-loss is disabled.{RST}\n")
    elif args.stats:
        from metrics.reporting import print_stats
        print_stats()
    elif args.positions:
        from metrics.reporting import print_positions
        print_positions()
    elif args.history:
        from metrics.reporting import print_history
        print_history()
    elif args.cities:
        from metrics.reporting import print_cities
        print_cities()
    elif args.calibration:
        from metrics.reporting import print_calibration
        from metrics.calibration import update_shrinkage_factors
        print_calibration()
        stored = update_shrinkage_factors()
        if stored:
            print(f"{G}Shrinkage factors updated:{RST}")
            for mt, sf in stored.items():
                print(f"  {mt}: {sf:.3f}")
        else:
            print(f"{DIM}Not enough trades per type to update shrinkage factors "
                  f"(need ≥15 per market type).{RST}")
    elif args.export_calibration:
        from metrics.calibration import export_calibration_csv
        filepath = args.export_calibration
        n = export_calibration_csv(filepath)
        print(f"{G}✓ Exported {n} rows to {filepath}{RST}")
    elif args.scrape_prices:
        from data.price_scraper import scrape_and_store_all_prices
        result = scrape_and_store_all_prices()
        print(f"\nDone — {result['markets_scraped']} markets scraped, "
              f"{result['total_points']:,} price points stored, "
              f"{result['errors']} errors")
    else:
        parser.print_help()
        print(f"\n{Y}Start with: python main.py --backfill{RST}")
        print(f"{Y}Then:       python main.py --scan{RST}")


if __name__ == "__main__":
    main()

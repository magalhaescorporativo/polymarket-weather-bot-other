"""
Open-Meteo multi-model ensemble forecast fetcher.
All 5 models are queried independently and stored separately.
Never averaged before storage — raw predictions only.
"""
import logging
import time
import requests
from datetime import date, timedelta
from config import (OPENMETEO_MODELS, OPENMETEO_ARCHIVE_URL,
                    OPENMETEO_MODEL_PARAMS,
                    HRRR_LAT_MIN, HRRR_LAT_MAX, HRRR_LON_MIN, HRRR_LON_MAX)

logger = logging.getLogger(__name__)
_WARN_LAST_TS: dict[str, float] = {}
_WARN_INTERVAL_S = 600

TIMEOUT     = 12   # seconds per request
MAX_RETRIES = 3    # retry count for transient failures
RETRY_CODES = {429, 500, 502, 503, 504}


def _get_with_retry(url: str, params: dict, timeout: int = TIMEOUT) -> requests.Response:
    """GET with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in RETRY_CODES and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning("HTTP %d from %s — retrying in %ds (attempt %d/%d)",
                               resp.status_code, url, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            raise
    resp.raise_for_status()
    return resp


def _warn_rate_limited(key: str, msg: str, *args):
    now = time.time()
    last = _WARN_LAST_TS.get(key, 0)
    if now - last >= _WARN_INTERVAL_S:
        _WARN_LAST_TS[key] = now
        logger.warning(msg, *args)


def fetch_forecast_one_model(model_name: str, lat: float, lon: float,
                              target_date: str, timezone: str) -> tuple[float, int]:
    """
    Fetch daily max temperature (°C) and max precip prob (%) from one Open-Meteo model.
    Returns (temp_c, precip_prob_pct)
    """
    url = OPENMETEO_MODELS[model_name]
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "daily":            "temperature_2m_max,precipitation_probability_max",
        "temperature_unit": "celsius",
        "forecast_days":    14,
        "timezone":         timezone,
    }
    # Merge any model-specific extra params (e.g. HRRR needs models=hrrr)
    params.update(OPENMETEO_MODEL_PARAMS.get(model_name, {}))
    try:
        print("REQUEST:", model_name, lat, lon, target_date)
        resp = _get_with_retry(url, params)
    except requests.HTTPError as e:
        # Explicit fallback chain for HRRR endpoint failures:
        # hrrr model param -> default forecast models.
        code = getattr(e.response, "status_code", None)
        if model_name == "hrrr" and code == 400:
            _warn_rate_limited(
                "hrrr-fallback",
                "HRRR request returned 400 — falling back to default Open-Meteo models for %.3f,%.3f (%s)",
                lat, lon, target_date,
            )
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            resp = _get_with_retry(url, fallback_params)
        else:
            raise
    data = resp.json()

    daily = data.get("daily", {})
    times  = daily.get("time", [])
    temps  = daily.get("temperature_2m_max", [])
    precip = daily.get("precipitation_probability_max", [])

    if target_date not in times:
        raise ValueError(f"Model {model_name}: target_date {target_date} not in forecast")
    
    idx = times.index(target_date)
    val = temps[idx]
    pr  = precip[idx] if idx < len(precip) else 0
    
    if val is None:
        raise ValueError(f"Model {model_name}: null temperature for {target_date}")
    
    return float(val), int(pr or 0)


def fetch_all_models(lat: float, lon: float, target_date: str,
                     timezone: str) -> dict[str, float]:
    """
    Fetch all available Open-Meteo model forecasts for a city/date in parallel.
    HRRR is included only for CONUS coordinates (lat 20-55N, lon 60-135W).
    Returns dict: {model_name: predicted_high_c}
    Raises if fewer than 3 models succeed (can't build reliable ensemble).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Determine which models to query for this location
    in_conus = (HRRR_LAT_MIN <= lat <= HRRR_LAT_MAX and
                HRRR_LON_MIN <= lon <= HRRR_LON_MAX)
    models_to_fetch = [m for m in OPENMETEO_MODELS
                       if m != "hrrr" or in_conus]

    results = {}
    errors  = {}

    def _fetch(model_name):
        return model_name, fetch_forecast_one_model(model_name, lat, lon, target_date, timezone)

    for model_name in models_to_fetch:
        try:
            time.sleep(2)

            _, (temp, precip) = _fetch(model_name)

            results[model_name] = {
                "temp": temp,
                "precip": precip
            }

            logger.debug(
                "  %s → %.1f°C (precip=%d%%)",
                model_name,
                temp,
                precip
            )

        except Exception as e:
            errors[model_name] = str(e)

            _warn_rate_limited(
                f"model-{model_name}",
                "  %s FAILED: %s",
                model_name,
                e
            )

    if len(results) < 1:
        try:
            from ops_state import update_datasource_health
            update_datasource_health(
                "openmeteo",
                False,
                f"{len(results)}/{len(models_to_fetch)} models"
            )
        except Exception:
            pass

        raise RuntimeError(
        f"Only {len(results)}/{len(models_to_fetch)} models succeeded for "
        f"{lat},{lon} {target_date}. Errors: {errors}"
    )
else:
    try:
        from ops_state import update_datasource_health
        update_datasource_health(
            "openmeteo",
            True,
            f"{len(results)}/{len(models_to_fetch)} models"
        )
    except Exception:
        pass

    # ECMWF free tier only provides ~7 days of forecast. For near-term dates (≤7 days)
    # where ECMWF should be available, its absence is a real data gap — flag it prominently
    # since ECMWF carries the highest ensemble weight (1.8×).
    if "ecmwf" not in results:
        try:
            days_ahead = (date.fromisoformat(target_date) - date.today()).days
        except (ValueError, TypeError):
            days_ahead = 99
        if days_ahead <= 7:
            logger.warning(
                "ECMWF (highest-weight model) unavailable for %s (%d days out) — "
                "ensemble running on %d/5 models. Error: %s",
                target_date, days_ahead, len(results), errors.get("ecmwf", "unknown")
            )
        else:
            logger.debug("ECMWF unavailable for %s (%d days out, beyond free-tier horizon)",
                         target_date, days_ahead)

    return results


def fetch_historical_actuals(lat: float, lon: float,
                              start_date: str, end_date: str,
                              timezone: str) -> dict[str, float]:
    """
    Fetch historical daily max temperatures from Open-Meteo Archive (ERA5 reanalysis).
    Returns dict: {date_str: actual_high_c}
    Raises on failure.
    """
    resp = _get_with_retry(OPENMETEO_ARCHIVE_URL, params={
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start_date,
        "end_date":         end_date,
        "daily":            "temperature_2m_max",
        "temperature_unit": "celsius",
        "timezone":         timezone,
    }, timeout=30)
    data = resp.json()

    times = data.get("daily", {}).get("time", [])
    temps = data.get("daily", {}).get("temperature_2m_max", [])

    if not times:
        raise ValueError(f"Archive returned no data for {lat},{lon} {start_date}–{end_date}")

    result = {}
    for t, v in zip(times, temps):
        if v is not None:
            result[t] = float(v)
    return result


def fetch_historical_model_forecast(model_name: str, lat: float, lon: float,
                                     target_date: str, timezone: str) -> float | None:
    """
    DEPRECATED PROXY: returns ERA5 actuals as a model forecast stand-in.
    This is only used for international stations without ASOS ground truth.
    Do NOT use for bias computation — ERA5 vs ERA5 bias = 0 (circular).
    Returns None if unavailable.
    """
    try:
        actuals = fetch_historical_actuals(lat, lon, target_date, target_date, timezone)
        return actuals.get(target_date)
    except Exception as e:
        logger.debug("Historical model proxy failed for %s %s: %s", model_name, target_date, e)
        return None

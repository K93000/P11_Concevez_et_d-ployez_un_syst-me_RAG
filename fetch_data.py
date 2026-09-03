import json
import os
from datetime import datetime, timedelta, timezone

# ============================================================
# CONFIGURATION & TEMPORAL ANCHOR
# ============================================================

INPUT_FILE = "data"
OUTPUT_FILE = os.path.join("data", "events_93_raw.json")
TARGET_DEPARTMENT = "93"

NOW = datetime.now(timezone.utc)
ONE_YEAR_AGO = NOW - timedelta(days=365)


# ============================================================
# CHARGEMENT DU FICHIER SOURCE LOCAL
# ============================================================

def load_openagenda_file():
    """Charge le fichier JSON brut téléchargé localement ('data' ou 'data.json')."""
    possible_files = [INPUT_FILE, f"{INPUT_FILE}.json"]
    file_path = next((p for p in possible_files if os.path.exists(p)), None)

    if file_path is None:
        raise FileNotFoundError(
            "❌ Aucun fichier OpenAgenda trouvé. Place 'data' ou 'data.json' à la racine."
        )

    print(f"📂 Fichier source identifié : {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = (
            data.get("results")
            or data.get("events")
            or data.get("records")
            or []
        )
    else:
        raise ValueError("❌ Structure JSON OpenAgenda non reconnue.")

    print(f"📦 Événements bruts chargés : {len(records):,}".replace(",", " "))
    return records


# ============================================================
# FILTRAGE GÉOGRAPHIQUE (93)
# ============================================================

def is_in_department_93(event):
    """Vérifie le rattachement à la Seine-Saint-Denis via code postal ou département."""
    location = event.get("location", {})
    if not isinstance(location, dict):
        location = {}

    postal_code = str(
        event.get("location_postalcode")
        or event.get("postalcode")
        or event.get("postalCode")
        or location.get("postalCode")
        or ""
    ).strip()

    dept = str(
        event.get("location_department")
        or event.get("department")
        or location.get("department")
        or ""
    ).strip()

    return postal_code.startswith(TARGET_DEPARTMENT) or TARGET_DEPARTMENT in dept


# ============================================================
# PARSING & VALIDATION DES DATES
# ============================================================

def extract_dates(event):
    """Extrait les dates événementielles réelles."""
    first_date = (
        event.get("firstdate_begin")
        or event.get("firstdate")
        or event.get("firstDate")
        or event.get("start")
        or ""
    )
    last_date = (
        event.get("lastdate_end")
        or event.get("lastdate")
        or event.get("lastDate")
        or event.get("end")
        or first_date
    )
    return first_date, last_date


def parse_date(date_value):
    """Convertit une chaîne ISO en datetime UTC."""
    if not date_value:
        return None
    try:
        value = str(date_value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def is_recent_event(event, start_date, now):
    """Vérifie l'appartenance à la fenêtre [NOW - 365j, NOW] et la cohérence chronologique."""
    first_date, last_date = extract_dates(event)
    start_dt = parse_date(first_date)
    end_dt = parse_date(last_date)

    if start_dt is None:
        return False
    if end_dt is None:
        end_dt = start_dt

    return (
        start_date <= start_dt <= now
        and end_dt <= now
        and end_dt >= start_dt
    )


def get_event_id(event):
    """Extrait un identifiant unique robuste."""
    return str(
        event.get("uid")
        or event.get("id")
        or event.get("recordid")
        or ""
    ).strip()


# ============================================================
# PIPELINE DE TRAITEMENT
# ============================================================

def fetch_and_filter_events():
    print("=" * 65)
    print("      OPENAGENDA → EXTRACTION STRICTE DU 93 (PASSÉ RÉCENT)")
    print("=" * 65)
    print(f"📅 Fenêtre : {ONE_YEAR_AGO.strftime('%Y-%m-%d')} → {NOW.strftime('%Y-%m-%d')}")
    print("📍 Périmètre : Seine-Saint-Denis (93)\n")

    raw_events = load_openagenda_file()
    filtered_events = []
    seen_ids = set()

    skipped_dept = skipped_dates = skipped_dups = skipped_noid = 0

    for event in raw_events:
        event_id = get_event_id(event)

        if not event_id:
            skipped_noid += 1
            continue
        if event_id in seen_ids:
            skipped_dups += 1
            continue
        if not is_in_department_93(event):
            skipped_dept += 1
            continue
        if not is_recent_event(event, ONE_YEAR_AGO, NOW):
            skipped_dates += 1
            continue

        seen_ids.add(event_id)
        filtered_events.append(event)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_events, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("📊 RÉSULTAT DE L'EXTRACTION")
    print("=" * 65)
    print(f"Événements bruts            : {len(raw_events):,}".replace(",", " "))
    print(f"Événements 93 conservés     : {len(filtered_events):,}".replace(",", " "))
    print(f"Hors 93                     : {skipped_dept:,}".replace(",", " "))
    print(f"Hors période / Incohérents  : {skipped_dates:,}".replace(",", " "))
    print(f"Doublons ignorés            : {skipped_dups:,}".replace(",", " "))
    print(f"Sans identifiant            : {skipped_noid:,}".replace(",", " "))
    print(f"\n💾 Fichier généré : {OUTPUT_FILE}")
    print("=" * 65)

    return filtered_events


if __name__ == "__main__":
    fetch_and_filter_events()
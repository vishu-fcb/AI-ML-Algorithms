"""
BeyondTech - Parts & Workshop Marketplace
Parts pricing (static estimates) + nearby workshops via Google Maps Places API.
Workshop results are cached per vehicle for the entire Streamlit session to
minimise API calls — fetched once on first open, reused on every subsequent re-run.
"""

import os
import urllib.parse
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Parts catalogue — static estimates, no API calls
# ---------------------------------------------------------------------------
PARTS_CATALOG = [
    {
        "part": "Tires (set of 4)",
        "EV":     {"price_low": 120, "price_high": 280, "query": "electric car tires set of 4"},
        "ICE":    {"price_low":  80, "price_high": 220, "query": "car tires set of 4"},
        "Hybrid": {"price_low":  90, "price_high": 240, "query": "hybrid car tires set of 4"},
    },
    {
        "part": "Brake Pads (axle set)",
        "EV":     {"price_low": 35, "price_high": 110, "query": "electric vehicle brake pads"},
        "ICE":    {"price_low": 25, "price_high":  90, "query": "car brake pads axle set"},
        "Hybrid": {"price_low": 30, "price_high": 100, "query": "hybrid car brake pads"},
    },
    {
        "part": "Battery Health Assessment",
        "EV":     {"price_low": 50, "price_high": 150, "query": "EV battery health diagnostic service"},
        "ICE":    None,
        "Hybrid": {"price_low": 50, "price_high": 150, "query": "hybrid battery health diagnostic"},
    },
    {
        "part": "Engine Oil + Filter",
        "EV":     None,
        "ICE":    {"price_low": 20, "price_high":  70, "query": "engine oil filter change kit"},
        "Hybrid": {"price_low": 20, "price_high":  65, "query": "hybrid engine oil filter kit"},
    },
    {
        "part": "Air Filter",
        "EV":     {"price_low": 15, "price_high":  40, "query": "cabin air filter electric vehicle"},
        "ICE":    {"price_low": 10, "price_high":  35, "query": "engine air filter replacement"},
        "Hybrid": {"price_low": 12, "price_high":  38, "query": "hybrid air filter replacement"},
    },
    {
        "part": "Coolant (5L)",
        "EV":     {"price_low": 20, "price_high":  55, "query": "electric vehicle coolant 5L"},
        "ICE":    {"price_low": 15, "price_high":  45, "query": "engine coolant antifreeze 5L"},
        "Hybrid": {"price_low": 18, "price_high":  50, "query": "hybrid coolant antifreeze 5L"},
    },
    {
        "part": "Transmission Fluid",
        "EV":     None,
        "ICE":    {"price_low": 30, "price_high":  80, "query": "automatic transmission fluid change"},
        "Hybrid": {"price_low": 30, "price_high":  75, "query": "hybrid transmission fluid"},
    },
]

# ---------------------------------------------------------------------------
# Vehicle → home city + extra cities for country-wide workshop coverage
# ---------------------------------------------------------------------------
VEHICLE_CITY_MAP = {
    "Tesla Model 3 Performance": {
        "home": "Amsterdam, Netherlands",
        "extra_cities": ["Rotterdam, Netherlands", "The Hague, Netherlands", "Utrecht, Netherlands", "Eindhoven, Netherlands"],
    },
    "BMW i4 eDrive40": {
        "home": "Berlin, Germany",
        "extra_cities": ["Hamburg, Germany", "Munich, Germany", "Frankfurt, Germany", "Cologne, Germany", "Stuttgart, Germany"],
    },
    "Mercedes EQS 450+": {
        "home": "Stuttgart, Germany",
        "extra_cities": ["Munich, Germany", "Frankfurt, Germany", "Cologne, Germany", "Hamburg, Germany", "Berlin, Germany"],
    },
    "Porsche Taycan Turbo S": {
        "home": "Frankfurt, Germany",
        "extra_cities": ["Stuttgart, Germany", "Munich, Germany", "Berlin, Germany", "Hamburg, Germany", "Cologne, Germany"],
    },
    "Ford F-150 XL": {
        "home": "Munich, Germany",
        "extra_cities": ["Frankfurt, Germany", "Stuttgart, Germany", "Nuremberg, Germany", "Augsburg, Germany"],
    },
    "BMW 330i xDrive": {
        "home": "Munich, Germany",
        "extra_cities": ["Frankfurt, Germany", "Stuttgart, Germany", "Nuremberg, Germany", "Augsburg, Germany"],
    },
    "Toyota Camry LE": {
        "home": "Hamburg, Germany",
        "extra_cities": ["Berlin, Germany", "Bremen, Germany", "Hanover, Germany", "Dortmund, Germany"],
    },
    "VW Golf GTI": {
        "home": "Berlin, Germany",
        "extra_cities": ["Hamburg, Germany", "Munich, Germany", "Frankfurt, Germany", "Cologne, Germany", "Stuttgart, Germany"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ebay_url(query: str) -> str:
    return f"https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote_plus(query)}"


@st.cache_resource
def get_gmaps_client():
    """
    Cached Google Maps client — instantiated once per Streamlit session.
    Returns None if the API key is missing or the library is unavailable.
    """
    if not GOOGLE_MAPS_API_KEY:
        return None
    try:
        import googlemaps
        return googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    except Exception:
        return None


@st.cache_data(show_spinner="Fetching nearby workshops…")
def _fetch_all_workshops(vehicle_name: str):
    """
    Geocode the vehicle's home city and fetch car_repair places within 5 km.
    Result is cached for the entire Streamlit session — the API is called
    only ONCE per vehicle, no matter how many times the panel re-renders.
    Returns a list of workshop dicts: {name, rating, address, lat, lng}.
    Also returns (home_lat, home_lng) for the map centre.
    """
    gmaps = get_gmaps_client()
    if not gmaps:
        return None, []

    config = VEHICLE_CITY_MAP.get(vehicle_name)
    if not config:
        return None, []

    def geocode(city):
        try:
            res = gmaps.geocode(city)
            if res:
                loc = res[0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            pass
        return None

    def places_near(lat, lng, radius=5000):
        try:
            res = gmaps.places_nearby(location=(lat, lng), radius=radius, type="car_repair")
            return res.get("results", [])
        except Exception:
            return []

    home_coords = geocode(config["home"])
    if not home_coords:
        return None, []

    seen = set()
    workshops = []

    for raw in places_near(*home_coords):
        try:
            lat = raw["geometry"]["location"]["lat"]
            lng = raw["geometry"]["location"]["lng"]
            key = (round(lat, 5), round(lng, 5))
            if key not in seen:
                seen.add(key)
                workshops.append({
                    "name": raw.get("name", "Workshop"),
                    "rating": raw.get("rating", "N/A"),
                    "address": raw.get("vicinity", ""),
                    "lat": lat, "lng": lng,
                })
        except (KeyError, TypeError):
            continue

    return home_coords, workshops


# ---------------------------------------------------------------------------
# Render functions called from app.py
# ---------------------------------------------------------------------------

def render_parts_table(vehicle_type: str):
    """Render a styled parts price table for the given vehicle type."""
    rows = []
    for item in PARTS_CATALOG:
        pricing = item.get(vehicle_type)
        if pricing is None:
            continue
        rows.append({
            "Part": item["part"],
            "Est. Price (EUR)": f"€{pricing['price_low']}–€{pricing['price_high']}",
            "Buy Online": f'<a href="{_ebay_url(pricing["query"])}" target="_blank">Search on eBay</a>',
        })

    if not rows:
        st.info("No parts information available for this vehicle type.")
        return

    header = "<tr><th>Part</th><th>Est. Price (EUR)</th><th>Buy Online</th></tr>"
    body = "".join(
        f"<tr><td>{r['Part']}</td><td>{r['Est. Price (EUR)']}</td><td>{r['Buy Online']}</td></tr>"
        for r in rows
    )
    st.markdown(f"""
    <style>
        .parts-table {{ width:100%; border-collapse:collapse; font-size:14px; margin-bottom:12px; }}
        .parts-table th {{ background:#1e1e3a; color:#aaa; padding:8px 12px; text-align:left; font-weight:600; border-bottom:1px solid #333; }}
        .parts-table td {{ padding:8px 12px; border-bottom:1px solid #2a2a2a; color:#e0e0e0; }}
        .parts-table tr:hover td {{ background:#1a1a2e; }}
        .parts-table a {{ color:#4fc3f7; text-decoration:none; }}
        .parts-table a:hover {{ text-decoration:underline; }}
    </style>
    <table class="parts-table">
        <thead>{header}</thead>
        <tbody>{body}</tbody>
    </table>
    <p style="font-size:11px; color:#666; margin-top:4px;">
        * Prices are estimated retail ranges in EUR. Actual prices vary by brand and supplier.
    </p>
    """, unsafe_allow_html=True)


def render_workshop_map(vehicle_name: str):
    """
    Build and return a folium Map using cached Places API data.
    The API is only called the first time this vehicle's map is requested.
    """
    import folium

    home_coords, workshops = _fetch_all_workshops(vehicle_name)
    if not home_coords:
        return None

    home_lat, home_lng = home_coords
    m = folium.Map(location=[home_lat, home_lng], zoom_start=13, tiles="CartoDB dark_matter")

    folium.Marker(
        location=[home_lat, home_lng],
        popup=folium.Popup(f"<b>{vehicle_name}</b><br>Home city", max_width=200),
        icon=folium.Icon(color="blue", icon="car", prefix="fa"),
    ).add_to(m)

    for w in workshops:
        directions_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&destination={w['lat']},{w['lng']}"
        )
        popup_html = (
            f"<b>{w['name']}</b><br>"
            f"Rating: {w['rating']} ★<br>"
            f"{w['address']}<br>"
            f'<a href="{directions_url}" target="_blank">Get Directions</a>'
        )
        folium.Marker(
            location=[w["lat"], w["lng"]],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="red", icon="wrench", prefix="fa"),
        ).add_to(m)

    return m

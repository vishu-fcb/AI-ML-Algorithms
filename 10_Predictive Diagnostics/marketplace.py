"""
BeyondTech - Parts & Workshop Marketplace
Provides parts pricing data and nearby workshop search via Google Maps Places API.
"""

import os
import urllib.parse

from dotenv import load_dotenv

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Parts catalogue
# Keys match the component names shown in recommendations.
# Prices are in EUR (estimated retail, single-unit).
# EV-only parts are marked N/A for ICE; ICE-only parts are marked N/A for EV.
# ---------------------------------------------------------------------------
PARTS_CATALOG = [
    {
        "part": "Tires (set of 4)",
        "component": "tire_tread",
        "EV":     {"price_low": 120, "price_high": 280, "query": "electric car tires set of 4"},
        "ICE":    {"price_low":  80, "price_high": 220, "query": "car tires set of 4"},
        "Hybrid": {"price_low":  90, "price_high": 240, "query": "hybrid car tires set of 4"},
    },
    {
        "part": "Brake Pads (axle set)",
        "component": "brake_thickness",
        "EV":     {"price_low": 35, "price_high": 110, "query": "electric vehicle brake pads"},
        "ICE":    {"price_low": 25, "price_high":  90, "query": "car brake pads axle set"},
        "Hybrid": {"price_low": 30, "price_high": 100, "query": "hybrid car brake pads"},
    },
    {
        "part": "Battery Health Assessment",
        "component": "battery_soh",
        "EV":     {"price_low": 50, "price_high": 150, "query": "EV battery health diagnostic service"},
        "ICE":    None,
        "Hybrid": {"price_low": 50, "price_high": 150, "query": "hybrid battery health diagnostic"},
    },
    {
        "part": "Engine Oil + Filter",
        "component": "oil_life",
        "EV":     None,
        "ICE":    {"price_low": 20, "price_high":  70, "query": "engine oil filter change kit"},
        "Hybrid": {"price_low": 20, "price_high":  65, "query": "hybrid engine oil filter kit"},
    },
    {
        "part": "Air Filter",
        "component": "air_filter_life",
        "EV":     {"price_low": 15, "price_high":  40, "query": "cabin air filter electric vehicle"},
        "ICE":    {"price_low": 10, "price_high":  35, "query": "engine air filter replacement"},
        "Hybrid": {"price_low": 12, "price_high":  38, "query": "hybrid air filter replacement"},
    },
    {
        "part": "Coolant (5L)",
        "component": "coolant_life",
        "EV":     {"price_low": 20, "price_high":  55, "query": "electric vehicle coolant 5L"},
        "ICE":    {"price_low": 15, "price_high":  45, "query": "engine coolant antifreeze 5L"},
        "Hybrid": {"price_low": 18, "price_high":  50, "query": "hybrid coolant antifreeze 5L"},
    },
    {
        "part": "Transmission Fluid",
        "component": "transmission_life",
        "EV":     None,
        "ICE":    {"price_low": 30, "price_high":  80, "query": "automatic transmission fluid change"},
        "Hybrid": {"price_low": 30, "price_high":  75, "query": "hybrid transmission fluid"},
    },
]

# ---------------------------------------------------------------------------
# Vehicle → city mapping (for Google Maps Places workshop search)
# ---------------------------------------------------------------------------
VEHICLE_CITY_MAP = {
    "Tesla Model 3 Performance": "Amsterdam, Netherlands",
    "BMW i4 eDrive40":           "Berlin, Germany",
    "Mercedes EQS 450+":         "Stuttgart, Germany",
    "Porsche Taycan Turbo S":    "Frankfurt, Germany",
    "Ford F-150 XL":             "Munich, Germany",
    "BMW 330i xDrive":           "Munich, Germany",
    "Toyota Camry LE":           "Hamburg, Germany",
    "VW Golf GTI":               "Berlin, Germany",
}


def render_parts_table(vehicle_type: str, critical_components: list):
    """
    Render a styled parts price table in Streamlit for the given vehicle type.
    Only shows rows where the component is in critical_components (or all if empty).
    """
    import streamlit as st

    rows = []
    for item in PARTS_CATALOG:
        pricing = item.get(vehicle_type)
        if pricing is None:
            continue  # N/A for this vehicle type
        # Only show parts that are relevant (component flagged or show all)
        rows.append({
            "Part": item["part"],
            "Est. Price (EUR)": f"€{pricing['price_low']}–€{pricing['price_high']}",
            "Buy Online": f'<a href="{_ebay_url(pricing["query"])}" target="_blank">Search on eBay</a>',
        })

    if not rows:
        st.info("No parts information available for this vehicle type.")
        return

    # Render as HTML table for clickable links
    header = "<tr><th>Part</th><th>Est. Price (EUR)</th><th>Buy Online</th></tr>"
    body = "".join(
        f"<tr><td>{r['Part']}</td><td>{r['Est. Price (EUR)']}</td><td>{r['Buy Online']}</td></tr>"
        for r in rows
    )
    table_html = f"""
    <style>
        .parts-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 12px;
        }}
        .parts-table th {{
            background: #1e1e3a;
            color: #aaa;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #333;
        }}
        .parts-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #2a2a2a;
            color: #e0e0e0;
        }}
        .parts-table tr:hover td {{
            background: #1a1a2e;
        }}
        .parts-table a {{
            color: #4fc3f7;
            text-decoration: none;
        }}
        .parts-table a:hover {{
            text-decoration: underline;
        }}
    </style>
    <table class="parts-table">
        <thead>{header}</thead>
        <tbody>{body}</tbody>
    </table>
    <p style="font-size:11px; color:#666; margin-top:4px;">
        * Prices are estimated retail ranges in EUR. Actual prices vary by brand and supplier.
    </p>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _ebay_url(query: str) -> str:
    return f"https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote_plus(query)}"


def get_gmaps_client():
    """Return a cached Google Maps client. Returns None if no API key is set."""
    if not GOOGLE_MAPS_API_KEY:
        return None
    try:
        import googlemaps
        return googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    except Exception:
        return None


def render_workshop_map(vehicle_name: str, gmaps_client):
    """
    Build and return a folium Map with nearby car-repair workshops.
    Returns None if geocoding or places search fails.
    """
    import folium

    city = VEHICLE_CITY_MAP.get(vehicle_name)
    if not city:
        return None

    try:
        geocode_result = gmaps_client.geocode(city)
        if not geocode_result:
            return None
        location = geocode_result[0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]
    except Exception:
        return None

    try:
        places_result = gmaps_client.places_nearby(
            location=(lat, lng),
            radius=5000,
            type="car_repair",
        )
    except Exception:
        return None

    m = folium.Map(location=[lat, lng], zoom_start=13, tiles="CartoDB dark_matter")

    # Centre marker
    folium.Marker(
        location=[lat, lng],
        popup=f"<b>{vehicle_name}</b><br>Home city: {city}",
        icon=folium.Icon(color="blue", icon="car", prefix="fa"),
    ).add_to(m)

    for place in places_result.get("results", []):
        try:
            p_lat = place["geometry"]["location"]["lat"]
            p_lng = place["geometry"]["location"]["lng"]
            name = place.get("name", "Workshop")
            rating = place.get("rating", "N/A")
            address = place.get("vicinity", "")
            directions_url = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&destination={p_lat},{p_lng}"
            )
            popup_html = (
                f"<b>{name}</b><br>"
                f"Rating: {rating} ★<br>"
                f"{address}<br>"
                f'<a href="{directions_url}" target="_blank">Get Directions</a>'
            )
            folium.Marker(
                location=[p_lat, p_lng],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color="red", icon="wrench", prefix="fa"),
            ).add_to(m)
        except (KeyError, TypeError):
            continue

    return m

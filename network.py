# ----------------------------------------------------------------------------
# Ports & hubs (lat, lon)
# ----------------------------------------------------------------------------
PORTS = {
    "CGP": {"name": "Chattogram, Bangladesh",        "lat": 22.3569, "lon": 91.7832, "type": "origin"},
    "CMB": {"name": "Colombo, Sri Lanka",             "lat": 6.9271,  "lon": 79.8612, "type": "hub"},
    "SIN": {"name": "Singapore",                      "lat": 1.2644,  "lon": 103.8220, "type": "hub"},
    "PKG": {"name": "Port Klang, Malaysia",            "lat": 3.0000,  "lon": 101.4000, "type": "hub"},
    "JEA": {"name": "Jebel Ali, UAE",                  "lat": 25.0118, "lon": 55.0618, "type": "hub"},
    "SUEZ":{"name": "Suez Canal",                      "lat": 30.0,    "lon": 32.5,    "type": "waypoint"},
    "RTM": {"name": "Rotterdam, Netherlands",          "lat": 51.9496, "lon": 4.1453,  "type": "hub"},
    "ANR": {"name": "Antwerp, Belgium",                "lat": 51.2705, "lon": 4.3411,  "type": "hub"},
    "FXT": {"name": "Felixstowe, UK",                  "lat": 51.9540, "lon": 1.3512,  "type": "destination_port"},
    "SOU": {"name": "Southampton, UK",                 "lat": 50.9097, "lon": -1.4044, "type": "destination_port"},
    "LGP": {"name": "London Gateway / Tilbury, UK",    "lat": 51.4700, "lon": 0.4700,  "type": "destination_port"},
}

# Final sea port + inland DC coordinates for each modeled UK distribution centre
DESTINATION_PORT = {
    "UK-DC01": "FXT",
    "UK-DC02": "SOU",
    "UK-DC03": "LGP",
}

DC_COORDS = {
    "UK-DC01": {"name": "Ipswich DC",  "lat": 52.0567, "lon": 1.1482},
    "UK-DC02": {"name": "Southampton DC", "lat": 50.9500, "lon": -1.3000},
    "UK-DC03": {"name": "Dagenham DC", "lat": 51.5390, "lon": 0.1450},
}

# ----------------------------------------------------------------------------
# Candidate routes (generic — the destination leg is resolved per-shipment)
# ----------------------------------------------------------------------------
ROUTES = {
    "colombo_express": {
        "name": "Colombo Express",
        "hubs": ["CGP", "CMB", "SUEZ"],
        "base_transit_days": 17,
        "cost_index": 1.00,
        "congestion_hub": "CMB",
        "blurb": "The mainstream lane — shortest, cheapest, but Colombo gets congested during "
                 "the SW monsoon (Jun–Sep) and pre-Christmas peak season (Nov–Dec).",
    },
    "singapore_transship": {
        "name": "Singapore Transship",
        "hubs": ["CGP", "SIN", "SUEZ", "RTM"],
        "base_transit_days": 21,
        "cost_index": 1.08,
        "congestion_hub": "SIN",
        "blurb": "An extra transshipment leg via Rotterdam adds days in normal conditions, but "
                 "Singapore's schedule reliability is high outside Chinese New Year (Jan–Feb).",
    },
    "port_klang_link": {
        "name": "Port Klang Link",
        "hubs": ["CGP", "PKG", "SUEZ", "ANR"],
        "base_transit_days": 19,
        "cost_index": 1.04,
        "congestion_hub": "PKG",
        "blurb": "A mid-cost alternative to Singapore with similar Jan–Feb congestion exposure "
                 "but slightly shorter dwell times most of the year.",
    },
    "jebel_ali_gulf": {
        "name": "Jebel Ali Gulf Route",
        "hubs": ["CGP", "JEA"],
        "base_transit_days": 15,
        "cost_index": 1.15,
        "congestion_hub": "JEA",
        "blurb": "The fastest and most expensive lane. Reliable most of the year, but Gulf "
                 "summer heat/storm operations (Jun–Aug) sharply increase variance.",
    },
}

# Monthly congestion multiplier per hub (1.0 = baseline). >1.2 materially raises
# both expected transit time and delay probability for routes through that hub.
CONGESTION_PROFILE = {
    "CMB": {1: 0.90, 2: 0.85, 3: 0.90, 4: 0.95, 5: 1.10, 6: 1.40,
            7: 1.50, 8: 1.45, 9: 1.30, 10: 1.05, 11: 1.35, 12: 1.40},
    "SIN": {1: 1.25, 2: 1.30, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.95,
            7: 1.00, 8: 1.00, 9: 1.00, 10: 1.05, 11: 1.10, 12: 1.15},
    "PKG": {1: 1.15, 2: 1.20, 3: 0.95, 4: 0.90, 5: 0.90, 6: 0.95,
            7: 0.95, 8: 0.95, 9: 0.95, 10: 1.00, 11: 1.05, 12: 1.10},
    "JEA": {1: 0.85, 2: 0.85, 3: 0.90, 4: 1.00, 5: 1.20, 6: 1.45,
            7: 1.55, 8: 1.50, 9: 1.20, 10: 0.95, 11: 0.85, 12: 0.90},
}


def route_waypoints(route_id, destination):
    """Ordered list of (lat, lon) points for drawing the route polyline,
    including the final UK sea port and the inland DC."""
    route = ROUTES[route_id]
    dest_port_code = DESTINATION_PORT[destination]
    codes = route["hubs"] + [dest_port_code]
    points = [[PORTS[c]["lat"], PORTS[c]["lon"]] for c in codes]
    dc = DC_COORDS[destination]
    points.append([dc["lat"], dc["lon"]])
    return points


def congestion_for(route_id, month):
    hub = ROUTES[route_id]["congestion_hub"]
    return CONGESTION_PROFILE[hub][month]


if __name__ == "__main__":
    for rid, r in ROUTES.items():
        print(rid, "->", route_waypoints(rid, "UK-DC01"))

from typing import Literal, Optional, Dict, List, Union, Tuple

import networkx as nx
import geopandas as gpd
import pandas as pd
import osmnx as ox



# --- TYPE DEFINITIONS BASED ON HBEFA 4.2 SCHEME ---
# LOS A-B maps to HBEFA ID 1 (Freeflow); LOS C-D to ID 2 (Heavy);
# LOS E to ID 3 (Saturated) (PDF Page 55, Table 6).
LOSClass = Literal["A", "B", "C", "D", "E", "F"]

# Vehicle categories as analyzed in energy consumption scatter plots
# (PDF Page 37, Figure 4).
VehicleType = Literal[
    "gasoline_pc",
    "diesel_pc",
    "diesel_hgv",
    "gasoline_mc",
    "ev_pc",
]

# Route types from OSM highway
ROUTE_TYPE_PRIORITY = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "service",
    "living_street",
    "residential",
    "unclassified",
]

RouteType = Literal[*ROUTE_TYPE_PRIORITY]


def get_los(
    maxspeed_kmh: float,
    avg_speed_kmh: float,
    format: Literal["hbefa", "LOS"] = "LOS",
) -> Union[int, str]:
    """Determine the HBEFA Level of Service (LOS) from speed data.

    This function implements the speed-based LOS classification approach
    described in the HBEFA 4.2 guidelines. Actual average speed is compared
    against threshold values associated with the posted speed limit.

    Args:
        maxspeed_kmh: Posted speed limit of the road segment in km/h.
        avg_speed_kmh: Actual average travel speed on the segment in km/h.
        format: Output format.

            - ``"hbefa"`` returns the numeric HBEFA LOS ID.
            - ``"LOS"`` returns the corresponding LOS letter class.

    Returns:
        Either an HBEFA LOS ID or an LOS letter classification.

        Numeric mapping:

        - ``1``: Freeflow
        - ``2``: Heavy
        - ``3``: Saturated
        - ``4``: Stop+go
        - ``5``: Stop+go II

        Letter mapping:

        - ``B``: Freeflow
        - ``D``: Heavy
        - ``E``: Saturated
        - ``F``: Stop+go / Gridlock
    """

    # Table 9: Suggested average speed thresholds (km/h)
    # Format:
    # speed_limit: [
    #     freeflow-heavy,
    #     heavy-saturated,
    #     saturated-stopgo,
    #     stopgo-stopgoII
    # ]
    los_thresholds: Dict[int, List[float]] = {
        30: [28.0, 20.2, 12.4, 7.6],
        40: [34.0, 25.3, 15.5, 8.7],
        50: [41.0, 29.0, 17.1, 9.2],
        60: [54.0, 37.9, 19.8, 9.2],
        70: [62.0, 45.1, 23.0, 10.1],
        80: [72.0, 52.2, 25.5, 10.5],
        90: [83.0, 60.0, 26.0, 11.0],
        100: [92.0, 66.3, 26.0, 11.0],
        110: [106.0, 77.1, 26.0, 11.0],
        120: [117.0, 87.1, 26.0, 11.0],
        130: [127.0, 95.3, 26.0, 11.0],
        140: [135.0, 108.6, 26.0, 11.0],
    }

    # Snap speed limit to nearest available threshold set.
    valid_maxspeed = min(
        los_thresholds.keys(),
        key=lambda x: abs(x - maxspeed_kmh),
    )

    t1, t2, t3, t4 = los_thresholds[valid_maxspeed]

    if avg_speed_kmh >= t1:
        hbefa_id = 1
    elif avg_speed_kmh >= t2:
        hbefa_id = 2
    elif avg_speed_kmh >= t3:
        hbefa_id = 3
    elif avg_speed_kmh >= t4:
        hbefa_id = 4
    else:
        hbefa_id = 5

    if format == "hbefa":
        return hbefa_id

    letter_mapping = {
        1: "B",
        2: "D",
        3: "E",
        4: "F",
        5: "F",
    }

    return letter_mapping[hbefa_id]

def hbefa_row(
    distance_m: float,
    avg_speed_kmh: float,
    intersection_dist_m: Optional[float] = None,
    gradient_pct: float = 0,
    route_type: Optional[
        Union[
            RouteType,
            list[RouteType],
            tuple[RouteType, ...],
        ]
    ] = None,
    vehicle_type: VehicleType = "gasoline_pc",
    los: Optional[Union[LOSClass,int]] = None,
    max_speed: Optional[float] = None,
) -> float:
    """Estimates hot operational CO2 emissions using HBEFA 4.2 Application Guidelines.

    Args:
        distance_m: Total travel distance of the road segment in meters.
        avg_speed_kmh: The actual average speed driven on the segment (km/h).
        intersection_dist_m: Mean distance between intersections used to infer Area. 
            (PDF Page 39, Section 3.1).
        gradient_pct: Average road gradient as a percentage (e.g., 4.0 for 4%).
        route_type: OSM-style road classification for hierarchy mapping (PDF Page 45, Table 4).
        vehicle_type: Vehicle category and technology.
        los: Traffic density level A-F, mapping to HBEFA IDs 1-5 or HBEFA level (PDF Page 55, Table 6).

    Returns:
        float: Total estimated hot operational CO2 emissions in kilograms.

    Justifications:
        - AREA (PDF Page 39, Section 3.1): 'Urban' (ID 2) is defined by infrastructure 
          obstacles like traffic lights. Intersections < 1500m serve as a proxy.
        - HIERARCHY (PDF Page 47, Section 4.3.1, Figure 8): Rural Distributor (ID 30) 
          and Local (ID 40) roads show 10-30% higher emissions than Primary-Nat (ID 20) 
          due to higher driving dynamics (stops every 1km vs every 10km).
        - URBAN PENALTY (PDF Page 41, Section 3.3; PDF Page 43, Figure 7): For passenger 
          cars, energy consumption is 7% to 20% higher on urban TS than rural ones 
          on distributor roads. A hierarchy-weighted range (10-20%) is applied.
        - MOTORWAY EFFICIENCY (PDF Page 41, Section 3.3; PDF Page 42, Figure 6): 
          Rural motorways (ID 10) are 2-3% less efficient than City motorways (ID 11) 
          due to higher-speed acceleration events.
        - HGV SENSITIVITY (PDF Page 41, Section 3.3): Energy consumption for heavy 
          vehicles (Urban buses/HGVs) increases by up to 28% in urban areas.
    """
    if intersection_dist_m is None:
        intersection_dist_m = distance_m 

    if isinstance(route_type, (list,tuple)):

        # Select the highest-priority route type appearing in the list.
        route_type = next(
            (
                rt
                for rt in ROUTE_TYPE_PRIORITY
                if rt in route_type
            ),
            None,
        )

    if los is None:
        if max_speed is None:
            los = "C" 
        else:
            los = get_los(max_speed,avg_speed_kmh) 

    # 1. AREA INFERENCE (PDF Page 39, Section 3.1)
    is_urban = intersection_dist_m < 1500

    # 2. CONSOLIDATED ROAD MAPPING (PDF Page 45, Table 4)
    # Mapping OSM tags to (Rural_HBEFA_Type, Urban_HBEFA_Type)
    road_mapping = {
        "motorway": ("Motorway-Nat", "Motorway-City"),      # IDs 10, 11
        "motorway_link": ("Primary-Nat", "Primary-City"),    # IDs 20, 21
        "trunk": ("Primary-Nat", "Primary-City"), 
        "trunk_link": ("Primary-Nat", "Primary-City"),
        "primary": ("Primary-Nat", "Primary-City"), 
        "primary_link": ("Primary-Nat", "Primary-City"),
        "secondary": ("Distributor-Rural", "Distributor-Urban"), # ID 30
        "secondary_link": ("Distributor-Rural", "Distributor-Urban"), 
        "tertiary": ("Local-Rural", "Local-Urban"),          # ID 40
        "tertiary_link": ("Local-Rural", "Local-Urban"),
        "residential": ("Access-Rural", "Access-Urban"),     # ID 50
        "living_street": ("Access-Rural", "Access-Urban"),
        "service": ("Access-Rural", "Access-Urban"),
    }

    # Hierarchy Fallback (PDF Page 47, Section 4.3.1)
    if route_type in road_mapping:
        rural_type, urban_type = road_mapping[route_type]
        # Lower values as source pdf to campture urban environments better
        if (urban_type == "Primary-City") and (intersection_dist_m < 500):
            urban_type = "Distributor-Urban"
        if (urban_type == "Distributor-Urban") and (intersection_dist_m < 250):
            urban_type = "Local-Urban"
    else:
        if is_urban: # Lower values as source pdf to campture urban environments better
            if intersection_dist_m >= 500:
                if avg_speed_kmh > 70:
                    urban_type = "Motorway-City"
                else:
                    urban_type = "Primary-City"
            elif intersection_dist_m >= 250:
                urban_type = "Distributor-Urban"
            else:
                urban_type = "Local-Urban" 
        else: 
            if intersection_dist_m >= 5000:
                if avg_speed_kmh > 90:
                    rural_type = "Motorway-Nat"
                else:
                    rural_type = "Primary-Nat"
            elif intersection_dist_m >= 1500:
                rural_type = "Distributor-Rural"
            else:
                rural_type = "Local-Rural" 
            

    hbefa_road_type = urban_type if is_urban else rural_type

    # 3. ROAD TYPE PENALTIES
    # BASELINE: Primary-Nat (ID 20) = 1.00.
    # Urban penalties are applied based on the 7-20% range for cars (PDF Page 41, Sec 3.3).
    ROAD_TYPE_PENALTIES = {
        # NATIONAL (RURAL) VARIANTS
        "Motorway-Nat": 0.98,       # 3% penalty over ID 11 (PDF Page 41, Sec 3.3)
        "Primary-Nat": 1.00,        # Functional Baseline (PDF Page 45, Table 4)
        "Distributor-Rural": 1.20,  # ~20% hierarchy increase (PDF Page 47, Figure 8)
        "Local-Rural": 1.30,        # ~30% hierarchy increase (PDF Page 47, Figure 8)
        "Access-Rural": 1.40,

        # CITY (URBAN) VARIANTS
        "Motorway-City": 0.95,      # Conservative efficiency baseline
        "Primary-City": 1.10,       # +10% Urban penalty (Fits 7-20% range, Sec 3.3)
        "Distributor-Urban": 1.38,  # Hierarchy 1.20 * Urban 1.15 penalty
        "Local-Urban": 1.56,        # Hierarchy 1.30 * Urban 1.20 penalty
        "Access-Urban": 1.68        # Hierarchy 1.40 * Urban 1.20 penalty
    }
    road_penalty = ROAD_TYPE_PENALTIES.get(hbefa_road_type, 1.20)

    # 4. BASE EMISSION FACTOR (g/km) (PDF Page 37, Figure 4)
    v = avg_speed_kmh
    if vehicle_type == "gasoline_pc":
        base_ef_gkm = (215 - 2.6 * v + 0.019 * v**2)
    elif vehicle_type == "diesel_pc":
        base_ef_gkm = (190 - 2.2 * v + 0.017 * v**2)
    elif vehicle_type == "diesel_hgv":
        base_ef_gkm = (1200 * (v**-0.35))
    elif vehicle_type == "gasoline_mc":
        base_ef_gkm = (110 - 2.8 * v + 0.025 * v**2) 
    else: # ev_pc (Tank-to-wheel is 0, PDF Page 39, Section 3.1)
        base_ef_gkm = 0.0

    # 5. LOS MULTIPLIER (PDF Page 66, Figure 16)
    # Stop+go II (LOS 5) index is ~3x larger than freeflow (100 vs 300).
    if isinstance(los,int):
        los_map = {1: 1.0, 2: 1.05, 3: 1.1, 4: 2.0, 5: 3.0}
    else:
        los_map = {"A": 1.0, "B": 1.05, "C": 1.1, "D": 1.15, "E": 1.5, "F": 2.0}

    los_multiplier = los_map.get(los.upper(), 1.05)

    # 6. GRADIENT ADJUSTMENT (PDF Page 71, Section 8.3; PDF Page 72, Figure 18)
    grad_sensitivities = {
        "gasoline_pc": 0.15, "diesel_pc": 0.15, "ev_pc": 0.15,
        "diesel_hgv": 0.25, # Higher HGV sensitivity (PDF Page 83, Figure 26)
        "gasoline_mc": 0.10
    }
    sensitivity = grad_sensitivities.get(vehicle_type, 0.15)
    gradient_multiplier = 1.0 + (max(0, gradient_pct) * sensitivity)

    # 7. HGV MASS-RESTART PENALTY (PDF Page 41, Section 3.3)
    # Urban buses/HGVs energy consumption increases by up to 28% in urban areas.
    if is_urban and vehicle_type == "diesel_hgv":
        road_penalty *= 1.10 # Adjusts penalty toward the 28% upper limit

    # FINAL CALCULATION (PDF Page 28, Section 1.2.1)
    total_ef = base_ef_gkm * los_multiplier * gradient_multiplier * road_penalty
    total_co2_kg = (total_ef * (distance_m / 1000)) / 1000

    return round(total_co2_kg, 4)

def route_hbefa(
    route_edges: gpd.GeoDataFrame,
    avg_speed_col: str = "avg_speed",
    gradient_col: Optional[str] = "gradient",
    route_type_col: Optional[str] = "highway",
    vehicle_type: VehicleType = "gasoline_pc",
    los_col: Optional[str] = None,
    maxspeed_col: Optional[str] = "maxspeed",
    length_col: str = "length",
    return_total:bool=True
) -> float|gpd.GeoDataFrame:
    """Estimate total route-level CO2 emissions using HBEFA 4.2 approximations.

    This function applies :func:`edge_hbefa` to each edge/segment of a route
    represented as a GeoDataFrame and aggregates the resulting operational
    emissions.

    The implementation is designed for OSM-derived routing networks where
    each row corresponds to a traversed edge with associated traffic and
    geometric properties.

    Args:
        route_edges: GeoDataFrame containing route edge attributes.
        avg_speed_col: Column containing actual average travel speed (km/h).
        gradient_col: Optional column containing edge gradient (%).
            If missing or ``None``, flat terrain is assumed.
        route_type_col: Optional OSM highway classification column.
        vehicle_type: Vehicle technology category used for emission modeling.
        los_col: Optional LOS column containing either:

            - LOS letters (A-F)
            - HBEFA LOS IDs (1-5)

            If omitted, LOS is estimated from speed and max speed.
        maxspeed_col: Optional posted speed limit column used for LOS
            estimation.
        length_col: Column containing edge length in meters.
        return_total: If False return co2 emissions for every edge in route_edges

    Returns:
        float: Total estimated route operational CO2 emissions in kilograms.

    Raises:
        KeyError: If required columns are missing from the GeoDataFrame.

    """

    required_columns = [length_col, avg_speed_col]

    for col in required_columns:
        if col not in route_edges.columns:
            raise KeyError(
                f"Required column '{col}' not found in route_edges."
            )

    # Gracefully disable optional columns if absent.
    if gradient_col is not None and gradient_col not in route_edges.columns:
        gradient_col = None

    if route_type_col is not None and route_type_col not in route_edges.columns:
        route_type_col = None

    if los_col is not None and los_col not in route_edges.columns:
        los_col = None

    if maxspeed_col is not None and maxspeed_col not in route_edges.columns:
        maxspeed_col = None

    total_emissions = route_edges.apply(
        lambda row: hbefa_row(
            distance_m=row[length_col],
            avg_speed_kmh=row[avg_speed_col],
            intersection_dist_m=row[length_col],
            gradient_pct=(
                0.0
                if gradient_col is None
                else row[gradient_col]
            ),
            route_type=(
                None
                if route_type_col is None
                else row[route_type_col]
            ),
            vehicle_type=vehicle_type,
            los=(
                None
                if los_col is None
                else row[los_col]
            ),
            max_speed=(
                None
                if maxspeed_col is None
                else row[maxspeed_col]
            ),
        ),
        axis=1,
    )
    if return_total:
        return round(float(total_emissions.sum()), 4)
    else:
        return total_emissions
    
def isochrone_co2(
    G: Union[
        nx.MultiDiGraph,
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame],
    ],
    path_column: str,
    co2_col: str = "co2",
    edge_co2_col: str = "co2",
    travel_time_col: str = "length",
):
    """
    Vectorized CO2 aggregation over precomputed node paths.

    Each node must contain:
        node[path_column] = [n0, n1, n2, ...]

    CO2 is computed by selecting, for each edge (u, v),
    the edge with minimum travel_time_col in the edges GDF.

    Returns SAME format as input.
    """

    # ============================================================
    # PRESERVE INPUT FORMAT
    # ============================================================

    input_is_graph = isinstance(G, nx.MultiDiGraph)
    input_is_tuple_2 = isinstance(G, tuple) and len(G) == 2
    input_is_tuple_3 = isinstance(G, tuple) and len(G) == 3

    # ============================================================
    # GRAPH UNPACKING
    # ============================================================

    if input_is_tuple_3:
        G, nodes, edges = G
    elif input_is_tuple_2:
        nodes, edges = G
        G = ox.graph_from_gdfs(nodes, edges)
    else:
        nodes, edges = ox.graph_to_gdfs(G)

    # ensure clean edge format
    if "u" not in edges.columns or "v" not in edges.columns:
        edges = edges.reset_index()

    # ============================================================
    # EDGE TABLE (FAST VECTOR LOOKUP)
    # ============================================================

    edge_lookup = (
        edges[["u", "v", travel_time_col, edge_co2_col]]
        .sort_values(travel_time_col)
        .drop_duplicates(["u", "v"])
        .set_index(["u", "v"])
    )

    # ============================================================
    # EXTRACT PATHS (VECTORISED)
    # ============================================================

    path_series = nodes[path_column].dropna()

    df = path_series.explode().reset_index()
    df.columns = ["node", "path_node"]

    # create next step in path
    df["next_node"] = df.groupby("node")["path_node"].shift(-1)
    df = df.dropna(subset=["next_node"])

    df["u"] = df["path_node"]
    df["v"] = df["next_node"]

    # ============================================================
    # EDGE JOIN (VECTORISED)
    # ============================================================

    df = df.join(edge_lookup, on=["u", "v"])

    # ============================================================
    # CO2 AGGREGATION (VECTORISED)
    # ============================================================

    co2_series = df.groupby("node")[edge_co2_col].sum()

    # assign to nodes
    nodes[co2_col] = nodes.index.map(co2_series).fillna(0)

    # ============================================================
    # WRITE BACK INTO GRAPH IF NEEDED
    # ============================================================

    if input_is_graph:
        nx.set_node_attributes(G, co2_series.to_dict(), co2_col)
        return G

    if input_is_tuple_2:
        return nodes, edges

    if input_is_tuple_3:
        return G, nodes, edges

    return G
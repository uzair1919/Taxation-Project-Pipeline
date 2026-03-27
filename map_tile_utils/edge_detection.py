import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import json
import os
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
from map_tile_utils.config import (PLOT_MIN_AREA, PLOT_MAX_AREA, CLUSTER_THRESHOLD, 
                                    NOISE_THRESHOLD, MAX_ASPECT_RATIO, MIN_COMPACTNESS, MIN_VERTICES,
                                    EROSION_SIZE, GRAY_LIMIT, BLUE_LIMIT, BRIGHTNESS_LIMIT, BLUE_HUE_RANGE)
# ===================== HIGH-PRECISION PLOT EXTRACTOR =====================

class PlotExtractorCV:
    def __init__(self, image_path, geo_bounds):
        self.image_path = image_path
        self.geo_bounds = geo_bounds
        self.min_area = PLOT_MIN_AREA
        self.max_area = PLOT_MAX_AREA
        self.cluster_threshold = CLUSTER_THRESHOLD
        self.noise_threshold = NOISE_THRESHOLD
        self.max_aspect_ratio = MAX_ASPECT_RATIO
        self.min_compactness = MIN_COMPACTNESS
        self.min_vertices = MIN_VERTICES
        self.erosion_size = EROSION_SIZE
        self.gray_limit = GRAY_LIMIT
        self.blue_sat_limit = BLUE_LIMIT
        self.brightness_limit = BRIGHTNESS_LIMIT
        self.blue_hue_range = BLUE_HUE_RANGE

        
        # Load Image
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.h, self.w = self.img.shape[:2]

        # Calculate Degrees per Pixel
        self.lat_per_px = (geo_bounds['north'] - geo_bounds['south']) / self.h
        self.lon_per_px = (geo_bounds['east'] - geo_bounds['west']) / self.w

    def pixel_to_geo(self, x, y):
        lat = self.geo_bounds['north'] - y * self.lat_per_px
        lon = self.geo_bounds['west'] + x * self.lon_per_px
        return lat, lon

    def get_sealed_mask(self):
        """Processes the image to seal black boundaries and isolate plots."""
        # 1. Grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # 2. Contrast Stretching (LUT) 
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 100, 210, 255]
        lut = np.interp(np.arange(256), xp, fp).astype(np.uint8)
        enhanced = cv2.LUT(gray, lut)

        # 3. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 13, 5
        )

        # 4. Bridge the Gaps (Healing)
        kernel = np.ones((3, 3), np.uint8)
        healed_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 5. Invert: Plots become white islands on black background
        mask = cv2.bitwise_not(healed_lines)
        
        return mask

    def is_road_color(self, contour):

        # 1. Create mask and Erode
        # This prevents the black plot outlines from ruining the color sample
        contour_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        kernel = np.ones(self.erosion_size, np.uint8)
        contour_mask = cv2.erode(contour_mask, kernel, iterations=1)

        if cv2.countNonZero(contour_mask) == 0:
            return False # Shape too small to sample
        
        # 2. Extract HSV pixels
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        pixels = hsv[contour_mask == 255]

        # 3. Calculate MEDIANS (Robust to text/plot numbers)
        med_h = float(np.median(pixels[:, 0]))
        med_s = float(np.median(pixels[:, 1]))
        med_v = float(np.median(pixels[:, 2]))

        # RULE 1: Neutral/Gray Road Test
        # If it has almost no color, it's a road.
        if med_s < self.gray_limit:
            return True

        # RULE 2: Light Blue/Periwinkle Road Test
        # Check if Hue is in blue range AND it's not too vibrant (not a plot)
        is_blue = self.blue_hue_range[0] <= med_h <= self.blue_hue_range[1]
        is_washed_out = med_s < self.blue_sat_limit
        is_bright = med_v > self.brightness_limit
        
        if is_blue and is_washed_out and is_bright:
            return True

        # If it fails both, it's a Plot
        return False

    def improve_plot_boundary(self, df):
        """
        clean_boundaries.py
        -------------------
        Cleans plot polygon boundaries corrupted by text-cutout artefacts.

        Algorithm (per plot)
        --------------------
        Each iteration finds the globally best shortcut across ALL possible (i, j)
        vertex pairs — the one that maximises area gain. It applies that shortcut,
        then restarts the search on the updated polygon. Stops when no shortcut
        gives a positive area gain, or the polygon reaches min_vertices.

        This is a non-greedy, exhaustive search: every candidate is evaluated
        before any change is made, so the best global move is always taken.
        """

        # ---------------------------------------------------------------------------
        # Geometry helpers
        # ---------------------------------------------------------------------------

        def get_polygon_area(coords):
            """Unsigned polygon area via Shoelace formula."""
            coords = np.atleast_2d(coords)
            if coords.shape[0] < 3 or coords.shape[1] != 2:
                return 0.0
            x, y = coords[:, 0], coords[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


        def _seg_intersect(p, r, q, s):
            """True if segment P+t*r and Q+u*s properly cross (interior only)."""
            denom = r[0] * s[1] - r[1] * s[0]
            if abs(denom) < 1e-20:
                return False
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / denom
            u = (qp[0] * r[1] - qp[1] * r[0]) / denom
            eps = 1e-9
            return eps < t < 1 - eps and eps < u < 1 - eps


        def _shortcut_creates_intersection(arr, i, j):
            """
            Returns True if the direct edge arr[i] -> arr[j] crosses any existing
            edge that lies outside the replaced segment [i..j].
            """
            n = len(arr)
            p = arr[i]
            r = arr[j] - arr[i]
            for k in range(n):
                kn = (k + 1) % n
                # Skip edges inside or adjacent to the replaced segment
                if i <= k < j or i < kn <= j:
                    continue
                if kn == i or k == j:
                    continue
                if _seg_intersect(p, r, arr[k], arr[kn] - arr[k]):
                    return True
            return False


        def find_first_self_intersection(coords):
            """Return (i, j, point) for the first crossing pair of edges, or None."""
            pts = np.array(coords, dtype=float)
            n = len(pts)
            for i in range(n):
                p = pts[i]
                r = pts[(i + 1) % n] - p
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    q = pts[j]
                    s = pts[(j + 1) % n] - q
                    denom = r[0] * s[1] - r[1] * s[0]
                    if abs(denom) < 1e-20:
                        continue
                    qp = q - p
                    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
                    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
                    eps = 1e-9
                    if eps < t < 1 - eps and eps < u < 1 - eps:
                        return i, j, p + t * r
            return None


        def fix_self_intersections(coords, max_iter=40):
            """Split at crossing points, keep the larger (outer) ring."""
            coords = list(coords)
            for _ in range(max_iter):
                hit = find_first_self_intersection(coords)
                if hit is None:
                    break
                i, j, X = hit
                ring_a = coords[:i + 1] + [(X[0], X[1])] + coords[j + 1:]
                ring_b = coords[i + 1:j + 1] + [(X[0], X[1])]
                area_a = get_polygon_area(np.array(ring_a)) if len(ring_a) >= 3 else 0.0
                area_b = get_polygon_area(np.array(ring_b)) if len(ring_b) >= 3 else 0.0
                coords = ring_a if area_a >= area_b else ring_b
            return coords


        # ---------------------------------------------------------------------------
        # Main cleaning algorithm
        # ---------------------------------------------------------------------------

        def clean_plot_boundary(coords_list, min_vertices):
            """
            Clean a single plot polygon boundary by iteratively applying the
            globally best area-increasing shortcut.

            Each pass:
            1. Evaluate ALL possible (i, j) vertex pairs as shortcut candidates.
            2. A candidate is valid if:
                - j > i + 1  (non-adjacent, skips at least one vertex)
                - result has >= min_vertices vertices
                - the shortcut edge does not create a new self-intersection
                - area gain > 0
            3. Apply the single candidate with the highest area gain.
            4. Repeat on the updated polygon.
            5. Stop when no valid candidate exists.

            Parameters
            ----------
            coords_list : list of (lon, lat)
            min_vertices : int
                Minimum vertices to retain (default 7, as specified).

            Returns
            -------
            list of (lon, lat) tuples
            """
            # --- Sanitize ---
            clean = []
            for c in coords_list:
                if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 2:
                    try:
                        clean.append((float(c[0]), float(c[1])))
                    except (ValueError, TypeError):
                        pass
            if len(clean) < min_vertices:
                return clean

            # Deduplicate consecutive vertices (including wrap-around)
            deduped = [clean[0]]
            for pt in clean[1:]:
                if abs(pt[0] - deduped[-1][0]) > 1e-15 or abs(pt[1] - deduped[-1][1]) > 1e-15:
                    deduped.append(pt)
            while len(deduped) > 1 and (
                abs(deduped[0][0] - deduped[-1][0]) < 1e-15 and
                abs(deduped[0][1] - deduped[-1][1]) < 1e-15
            ):
                deduped.pop()

            if len(deduped) < min_vertices:
                return deduped

            coords = deduped

            # --- Iterative global-best shortcut ---
            improved = True
            while improved:
                improved = False
                n = len(coords)

                if n <= min_vertices:
                    break

                arr = np.array(coords, dtype=float)
                current_area = get_polygon_area(arr)

                best_gain = 0.0
                best_i, best_j = -1, -1

                # Evaluate ALL (i, j) pairs
                for i in range(n):
                    for j in range(i + 2, n):
                        # Skip the wrap-around adjacent pair
                        if i == 0 and j == n - 1:
                            continue

                        # Result must have enough vertices
                        # Shortcut removes vertices at positions i+1 ... j-1
                        result_n = n - (j - i - 1)
                        if result_n < min_vertices:
                            continue

                        # Compute area gain
                        candidate = coords[:i + 1] + coords[j:]
                        gain = get_polygon_area(np.array(candidate)) - current_area

                        # Must be positive gain
                        if gain <= 0:
                            continue

                        # Must not create a self-intersection
                        if _shortcut_creates_intersection(arr, i, j):
                            continue

                        # Best so far?
                        if gain > best_gain:
                            best_gain = gain
                            best_i, best_j = i, j

                # Apply the best shortcut found this pass
                if best_i != -1:
                    coords = coords[:best_i + 1] + coords[best_j:]
                    improved = True

            # Safety net: fix any remaining self-intersections
            fixed = fix_self_intersections(coords)
            if len(fixed) >= 3:
                coords = fixed

            return coords


        # ---------------------------------------------------------------------------
        # DataFrame interface (drop-in replacement)
        # ---------------------------------------------------------------------------

        def improve_plot_coordinates(df, min_vertices):
            """
            Process all plots in the dataframe, cleaning each polygon's boundary.

            Parameters
            ----------
            df : pd.DataFrame
                Must have columns: plot_id, vertex_index, latitude, longitude
            min_vertices : int
                Minimum vertices to retain per plot (default 7).

            Returns
            -------
            pd.DataFrame with cleaned coordinates and updated area_pixels.
            """
            improved_rows = []
            df = df.dropna(subset=['longitude', 'latitude'])

            for plot_id, group in df.groupby("plot_id"):
                group = group.sort_values("vertex_index")
                raw_coords = group[["longitude", "latitude"]].values.tolist()

                refined = clean_plot_boundary(raw_coords, min_vertices=min_vertices)

                if len(refined) < 3:
                    continue

                final_area = get_polygon_area(np.array(refined))

                for i, (lon, lat) in enumerate(refined):
                    improved_rows.append({
                        "plot_id": plot_id,
                        "vertex_index": i,
                        "latitude": lat,
                        "longitude": lon,
                        "area_pixels": int(final_area * 1e12)
                    })

            return pd.DataFrame(improved_rows)
        
        return improve_plot_coordinates(df, self.min_vertices)

    def remove_geometric_slivers(self, df):
        """
        Identifies and removes plots that are geometrically invalid (slivers, lines, noise).
        """
        df = df.sort_values(['plot_id', 'vertex_index'])
        valid_plot_ids = []

        for pid, group in df.groupby('plot_id'):
            if len(group) < 3:
                continue
                
            coords = list(zip(group['longitude'], group['latitude']))
            poly = Polygon(coords)
            
            # Basic validity check
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    poly = shapely.make_valid(poly)
            
            # Calculate bounding box geometry
            mrr = poly.minimum_rotated_rectangle
            if mrr.geom_type != 'Polygon':
                continue 

            # Aspect Ratio Calculation
            x, y = mrr.exterior.coords.xy
            sides = [np.sqrt((x[j]-x[j+1])**2 + (y[j]-y[j+1])**2) for j in range(3)]
            d1, d2 = sorted([sides[0], sides[1]]) # Short side, Long side
            
            aspect_ratio = d2 / d1 if d1 > 0 else 999
            
            # Compactness (Isoperimetric Quotient)
            # Higher is closer to a circle/square; Lower is a thin string
            compactness = (4 * np.pi * poly.area) / (poly.length ** 2) if poly.length > 0 else 0
            
            # Filter Logic
            if aspect_ratio <= self.max_aspect_ratio and compactness >= self.min_compactness:
                valid_plot_ids.append(pid)
                
        return df[df['plot_id'].isin(valid_plot_ids)]

    def filter_overlapping_plots(self, df):
        """
        Runs the original hierarchy logic to resolve containment and plot clusters.
        """
        # Reconstruction into GeoDataFrame for spatial operations
        poly_data = []
        for pid, group in df.groupby('plot_id'):
            coords = list(zip(group['longitude'], group['latitude']))
            poly = Polygon(coords)
            poly_data.append({'plot_id': pid, 'geometry': poly})
        
        if not poly_data:
            return df
            
        gdf = gpd.GeoDataFrame(poly_data)
        gdf['area'] = gdf.geometry.area
        gdf = gdf.sort_values('area', ascending=False).reset_index(drop=True)
        
        removed_ids = set()
        for i, row in gdf.iterrows():
            parent_id = row['plot_id']
            if parent_id in removed_ids: continue
            
            parent_geom = row['geometry']
            parent_area = row['area']
            
            # Check smaller plots against this larger 'parent'
            candidates = gdf[(gdf['area'] < parent_area) & (~gdf['plot_id'].isin(removed_ids))]
            
            captured_children_ids = []
            child_intersection_geoms = []
            
            for _, child in candidates.iterrows():
                try:
                    intersection_geom = child.geometry.intersection(parent_geom)
                    if intersection_geom.is_empty: continue
                    
                    # If child is mostly inside parent
                    if (intersection_geom.area / child.geometry.area) >= self.noise_threshold:
                        captured_children_ids.append(child.plot_id)
                        child_intersection_geoms.append(intersection_geom)
                except: continue

            num_children = len(captured_children_ids)
            
            # Logic: If many children fill the parent, the parent is redundant
            if num_children >= 3:
                try:
                    union_children = unary_union(child_intersection_geoms)
                    if (union_children.area / parent_area) >= self.cluster_threshold:
                        removed_ids.add(parent_id)
                except: continue
            # Logic: If only 1-2 children are inside, the children are likely noise/errors
            elif 0 < num_children < 3:
                for cid in captured_children_ids:
                    removed_ids.add(cid)

        final_keep_ids = set(gdf['plot_id']) - removed_ids
        return df[df['plot_id'].isin(final_keep_ids)]


    def extract_all(self):
        """Filters by area, aspect ratio, and color to separate plots from roads."""
        mask = self.get_sealed_mask()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        rows = []
        features = []
        plot_id = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # --- AREA FILTER ---
            if not (self.min_area < area < self.max_area):
                continue

            # --- ASPECT RATIO FILTER ---
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if not (0.2 < aspect_ratio < 5.0):
                continue

            # --- COLOR FILTER: skip roads ---
            if self.is_road_color(cnt):
                continue

            # Simplify shape to clean up pixelated edges
            epsilon = 0.006 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) < 3:
                continue
            
            corners_px = approx.reshape(-1, 2)
            geo_coords = []
            for px_x, px_y in corners_px:
                lat, lon = self.pixel_to_geo(int(px_x), int(px_y))
                geo_coords.append((lon, lat))
            
            geo_coords.append(geo_coords[0])  # Close loop

            for i, (lon, lat) in enumerate(geo_coords[:-1]):
                rows.append({
                    "plot_id": plot_id,
                    "vertex_index": i,
                    "latitude": lat,
                    "longitude": lon,
                    "area_pixels": int(area)
                })

            features.append({
                "type": "Feature",
                "properties": {"plot_id": plot_id, "area_px": int(area)},
                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
            })
            plot_id += 1

        df = pd.DataFrame(rows)
        geojson = {"type": "FeatureCollection", "features": features}
        return df, geojson, mask

    def extract_all(self):
        mask = self.get_sealed_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_rows = []
        temp_id = 1

        # --- 1. INITIAL EXTRACTION ---
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area): continue

            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.2 < (float(w) / h) < 5.0): continue
            if self.is_road_color(cnt): continue

            epsilon = 0.006 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3: continue
            
            corners_px = approx.reshape(-1, 2)
            for i, (px_x, px_y) in enumerate(corners_px):
                lat, lon = self.pixel_to_geo(int(px_x), int(px_y))
                raw_rows.append({
                    "plot_id": temp_id,
                    "vertex_index": i,
                    "latitude": lat,
                    "longitude": lon,
                    "area_pixels": int(area)
                })
            temp_id += 1

        df_raw = pd.DataFrame(raw_rows)
        if df_raw.empty:
            return df_raw, {"type": "FeatureCollection", "features": []}, mask

        # --- 2. THE TESTED WORKFLOW ---
        # a. Sliver Removal
        prerefined = self.remove_geometric_slivers(df_raw)
        # b. Vertex Improvement
        improved_coords = self.improve_plot_boundary(prerefined)
        # c. Overlap Resolution
        refined_coords = self.filter_overlapping_plots(improved_coords)

        # --- 3. FINAL FORMATTING (Matching Original Structure) ---
        final_rows = []
        final_features = []
        new_plot_id = 1

        # Group by old ID to re-assign a clean sequential ID
        for _, group in refined_coords.groupby("plot_id"):
            sorted_group = group.sort_values("vertex_index")
            area_px = int(sorted_group["area_pixels"].iloc[0])
            
            # Build coordinates for GeoJSON (tuples of lon, lat)
            geo_coords = []
            for _, row in sorted_group.iterrows():
                # Add to DataFrame rows
                final_rows.append({
                    "plot_id": new_plot_id,
                    "vertex_index": row["vertex_index"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "area_pixels": area_px
                })
                geo_coords.append((row["longitude"], row["latitude"]))
            
            # Close loop for GeoJSON (Original behavior)
            geo_coords.append(geo_coords[0])

            # Build GeoJSON Feature (Original keys: area_px)
            final_features.append({
                "type": "Feature",
                "properties": {"plot_id": new_plot_id, "area_px": area_px},
                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
            })
            new_plot_id += 1

        # Reconstruct final DF and GeoJSON
        df_final = pd.DataFrame(final_rows)
        geojson_final = {"type": "FeatureCollection", "features": final_features}

        return df_final, geojson_final, mask
# ===================== EXECUTION =====================
def exec(bounds_input, image_input):

    extractor = PlotExtractorCV(image_input, bounds_input)
    plot_coordinates, plot_geojson, debug_mask = extractor.extract_all()
    return plot_coordinates, plot_geojson

if __name__ == "__main__":
    exec()
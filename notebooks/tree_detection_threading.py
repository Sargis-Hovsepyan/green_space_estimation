# tree_detection_threading.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import geopandas as gpd
import pandas as pd

def detect_all_trees(ids, grid, buildings, edges, detect_func, max_workers=8):
    """
    Runs tree detection around buildings in parallel using threads.
    
    Parameters:
        ids (array-like): List of tile indexes.
        grid, buildings, edges (GeoDataFrames): Data inputs.
        detect_func (function): A function like detect_trees_around_buildings.
        max_workers (int): Number of threads.
    
    Returns:
        merged_trees_gdf, merged_buildings_gdf (GeoDataFrames)
    """

    def process_tile(tile_index):
        try:
            trees_gdf, blds, _, _ = detect_func(tile_index, grid, buildings, edges)
            return trees_gdf, blds
        except Exception as e:
            print(f"Error processing tile {tile_index}: {e}")
            return None, None

    all_tree_detections = []
    all_buildings = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tile, idx) for idx in ids]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Images Processed', dynamic_ncols=True):
            trees_gdf, blds = future.result()
            if trees_gdf is not None and blds is not None:
                all_tree_detections.append(trees_gdf)
                all_buildings.append(blds)

    merged_trees_gdf = gpd.GeoDataFrame(pd.concat(all_tree_detections, ignore_index=True), crs=grid.crs)
    merged_buildings_gdf = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True), crs=grid.crs)

    return merged_trees_gdf, merged_buildings_gdf
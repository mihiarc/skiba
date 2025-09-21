#!/usr/bin/env python
"""
Robust CSV Handler for Skiba
=============================

Handles various CSV formats and common issues from Excel exports,
ensuring reliable coordinate data extraction.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple, Union
import re


class CSVHandler:
    """Robust CSV handler for coordinate data with extensive validation."""

    # Possible column name variations
    LAT_COLUMNS = [
        'LAT', 'lat', 'Lat', 'latitude', 'Latitude', 'LATITUDE',
        'y', 'Y', 'lat_dd', 'LAT_DD', 'latitude_dd', 'LATITUDE_DD',
        'lat_deg', 'LAT_DEG', 'latitude_deg', 'LATITUDE_DEG',
        'POINT_Y', 'point_y', 'Y_COORD', 'y_coord', 'YCOORD', 'ycoord'
    ]

    LON_COLUMNS = [
        'LON', 'lon', 'Lon', 'longitude', 'Longitude', 'LONGITUDE',
        'long', 'Long', 'LONG', 'lng', 'Lng', 'LNG',
        'x', 'X', 'lon_dd', 'LON_DD', 'longitude_dd', 'LONGITUDE_DD',
        'lon_deg', 'LON_DEG', 'longitude_deg', 'LONGITUDE_DEG',
        'POINT_X', 'point_x', 'X_COORD', 'x_coord', 'XCOORD', 'xcoord'
    ]

    ID_COLUMNS = [
        'plot_ID', 'plot_id', 'plotID', 'plotId', 'PLOT_ID', 'PlotID',
        'ID', 'id', 'Id', 'identifier', 'Identifier', 'IDENTIFIER',
        'site_id', 'SITE_ID', 'SiteID', 'site_ID', 'point_id', 'POINT_ID',
        'location_id', 'LOCATION_ID', 'LocationID', 'loc_id', 'LOC_ID',
        'name', 'Name', 'NAME', 'site_name', 'SITE_NAME', 'plot_name', 'PLOT_NAME',
        'FID', 'fid', 'OBJECTID', 'objectid', 'OID', 'oid'
    ]

    def __init__(self, strict: bool = False):
        """
        Initialize CSV handler.

        Args:
            strict: If True, raises errors instead of warnings for data issues
        """
        self.strict = strict
        self.validation_report = {}

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by removing empty rows and columns.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Remove columns that are entirely NaN or unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')

        # Remove rows where all coordinate columns are NaN
        # We'll identify these columns first
        lat_col = self._find_column(df.columns, self.LAT_COLUMNS)
        lon_col = self._find_column(df.columns, self.LON_COLUMNS)

        if lat_col and lon_col:
            df = df.dropna(subset=[lat_col, lon_col], how='all')

        # Reset index after cleaning
        df = df.reset_index(drop=True)

        return df

    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """
        Find a column from a list of possible names.

        Args:
            columns: Available column names
            possible_names: List of possible column name variations

        Returns:
            Found column name or None
        """
        # First try exact match
        for name in possible_names:
            if name in columns:
                return name

        # Try case-insensitive match
        columns_lower = {col.lower(): col for col in columns}
        for name in possible_names:
            if name.lower() in columns_lower:
                return columns_lower[name.lower()]

        # Try partial match (e.g., "Latitude_WGS84" matches "latitude")
        for col in columns:
            for name in possible_names:
                if name.lower() in col.lower() or col.lower() in name.lower():
                    return col

        return None

    def validate_coordinates(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate coordinate values and fix common issues.

        Args:
            df: DataFrame with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            Tuple of (cleaned DataFrame, validation report)
        """
        report = {
            'total_rows': len(df),
            'invalid_coords': [],
            'out_of_range': [],
            'fixed_values': [],
            'dropped_rows': []
        }

        # Convert to numeric, handling various formats
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

        # Check for invalid coordinates
        invalid_mask = df[lat_col].isna() | df[lon_col].isna()
        if invalid_mask.any():
            invalid_indices = df[invalid_mask].index.tolist()
            report['invalid_coords'] = invalid_indices

            if self.strict:
                raise ValueError(f"Invalid coordinates at rows: {invalid_indices}")
            else:
                warnings.warn(f"Dropping {len(invalid_indices)} rows with invalid coordinates")
                df = df[~invalid_mask].copy()
                report['dropped_rows'] = invalid_indices

        # Validate coordinate ranges
        lat_out_of_range = (df[lat_col] < -90) | (df[lat_col] > 90)
        lon_out_of_range = (df[lon_col] < -180) | (df[lon_col] > 180)

        if lat_out_of_range.any():
            indices = df[lat_out_of_range].index.tolist()
            report['out_of_range'].extend([(idx, 'latitude') for idx in indices])

            # Check if coordinates might be swapped
            swapped = df[lat_out_of_range & ~lon_out_of_range].copy()
            if not swapped.empty:
                # Try swapping lat/lon
                temp = swapped[lat_col].copy()
                swapped[lat_col] = swapped[lon_col]
                swapped[lon_col] = temp

                # Check if swap fixes the issue
                lat_fixed = (swapped[lat_col] >= -90) & (swapped[lat_col] <= 90)
                lon_fixed = (swapped[lon_col] >= -180) & (swapped[lon_col] <= 180)

                if lat_fixed.all() and lon_fixed.all():
                    warnings.warn(f"Swapped lat/lon for {len(swapped)} rows that appeared reversed")
                    df.loc[swapped.index, [lat_col, lon_col]] = swapped[[lat_col, lon_col]]
                    report['fixed_values'].extend(swapped.index.tolist())

        if lon_out_of_range.any():
            indices = df[lon_out_of_range].index.tolist()
            report['out_of_range'].extend([(idx, 'longitude') for idx in indices])

        # Final check - remove any remaining out-of-range values
        final_invalid = (
            (df[lat_col] < -90) | (df[lat_col] > 90) |
            (df[lon_col] < -180) | (df[lon_col] > 180)
        )

        if final_invalid.any():
            invalid_indices = df[final_invalid].index.tolist()
            if self.strict:
                raise ValueError(f"Coordinates out of range at rows: {invalid_indices}")
            else:
                warnings.warn(f"Dropping {len(invalid_indices)} rows with out-of-range coordinates")
                df = df[~final_invalid].copy()
                report['dropped_rows'].extend(invalid_indices)

        report['valid_rows'] = len(df)
        return df, report

    def generate_ids(self, df: pd.DataFrame, id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Generate IDs if missing or ensure all rows have unique IDs.

        Args:
            df: DataFrame
            id_col: Name of ID column (will be created if None)

        Returns:
            DataFrame with valid IDs
        """
        if id_col is None:
            # Generate new ID column
            df['plot_ID'] = [f'POINT_{i+1:04d}' for i in range(len(df))]
            warnings.warn(f"No ID column found. Generated {len(df)} IDs (POINT_0001 to POINT_{len(df):04d})")
        else:
            # Check for missing IDs
            missing_ids = df[id_col].isna()
            if missing_ids.any():
                # Generate IDs for missing values
                num_missing = missing_ids.sum()
                new_ids = [f'POINT_{i+1:04d}' for i in range(num_missing)]
                df.loc[missing_ids, id_col] = new_ids
                warnings.warn(f"Generated {num_missing} IDs for rows with missing values")

            # Ensure IDs are unique
            duplicates = df[id_col].duplicated()
            if duplicates.any():
                dup_indices = df[duplicates].index
                for idx in dup_indices:
                    original_id = df.loc[idx, id_col]
                    df.loc[idx, id_col] = f"{original_id}_DUP{idx}"
                warnings.warn(f"Made {len(dup_indices)} duplicate IDs unique by adding suffix")

        return df

    def load_csv(self, filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load and clean CSV file with robust error handling.

        Args:
            filepath: Path to CSV file
            encoding: File encoding (tries others if this fails)

        Returns:
            Cleaned and validated DataFrame
        """
        # Try different encodings if needed
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        df = None
        for enc in encodings:
            try:
                # Use on_bad_lines='skip' to handle malformed rows
                df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # Try with different parsing options for problematic files
                try:
                    df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip', engine='python')
                    break
                except:
                    if enc == encodings[-1]:
                        raise ValueError(f"Could not read CSV file: {e}")

        if df is None:
            raise ValueError("Could not read CSV file with any common encoding")

        # Clean the dataframe
        df = self.clean_dataframe(df)

        if df.empty:
            raise ValueError("CSV file contains no valid data after cleaning")

        # Find required columns
        lat_col = self._find_column(df.columns, self.LAT_COLUMNS)
        lon_col = self._find_column(df.columns, self.LON_COLUMNS)
        id_col = self._find_column(df.columns, self.ID_COLUMNS)

        if not lat_col:
            raise ValueError(f"No latitude column found. Expected one of: {', '.join(self.LAT_COLUMNS[:5])}...")
        if not lon_col:
            raise ValueError(f"No longitude column found. Expected one of: {', '.join(self.LON_COLUMNS[:5])}...")

        # Validate coordinates
        df, validation_report = self.validate_coordinates(df, lat_col, lon_col)
        self.validation_report = validation_report

        # Handle IDs
        df = self.generate_ids(df, id_col)

        # Standardize column names
        rename_dict = {lat_col: 'LAT', lon_col: 'LON'}
        if id_col and id_col != 'plot_ID':
            rename_dict[id_col] = 'plot_ID'

        df = df.rename(columns=rename_dict)

        # Ensure we have the required columns
        required_cols = ['plot_ID', 'LAT', 'LON']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns after processing: {missing}")

        return df

    def get_summary(self) -> str:
        """Get a summary of the last validation."""
        if not self.validation_report:
            return "No validation performed yet"

        report = self.validation_report
        summary = []
        summary.append(f"Total rows processed: {report.get('total_rows', 0)}")
        summary.append(f"Valid rows: {report.get('valid_rows', 0)}")

        if report.get('dropped_rows'):
            summary.append(f"Dropped rows: {len(report['dropped_rows'])}")

        if report.get('fixed_values'):
            summary.append(f"Fixed coordinates (likely swapped): {len(report['fixed_values'])}")

        if report.get('invalid_coords'):
            summary.append(f"Invalid coordinates found: {len(report['invalid_coords'])}")

        if report.get('out_of_range'):
            summary.append(f"Out of range coordinates: {len(report['out_of_range'])}")

        return "\n".join(summary)


def process_csv_for_skiba(filepath: str, strict: bool = False) -> pd.DataFrame:
    """
    Convenience function to process a CSV file for use with skiba.

    Args:
        filepath: Path to CSV file
        strict: If True, raises errors instead of warnings

    Returns:
        Cleaned DataFrame ready for skiba
    """
    handler = CSVHandler(strict=strict)
    df = handler.load_csv(filepath)
    print("\n" + "="*50)
    print("CSV Processing Summary:")
    print("="*50)
    print(handler.get_summary())
    print("="*50 + "\n")
    return df
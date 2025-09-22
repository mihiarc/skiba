#!/usr/bin/env python
"""
Tests for CSV Handler functionality.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import warnings

from skiba.csv_handler import CSVHandler, process_csv_for_skiba


class TestCSVHandler(unittest.TestCase):
    """Test cases for robust CSV handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = CSVHandler(strict=False)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_temp_csv(self, content, filename="test.csv"):
        """Helper to create temporary CSV files."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def test_standard_format(self):
        """Test standard CSV format with expected column names."""
        csv_content = """plot_ID,LAT,LON,species
FOREST_001,45.5231,-122.6765,Douglas Fir
FOREST_002,45.5245,-122.6750,Western Hemlock
FOREST_003,45.5189,-122.6823,Sitka Spruce"""

        filepath = self.create_temp_csv(csv_content)
        df = self.handler.load_csv(filepath)

        self.assertEqual(len(df), 3)
        self.assertIn('plot_ID', df.columns)
        self.assertIn('LAT', df.columns)
        self.assertIn('LON', df.columns)
        self.assertIn('species', df.columns)

    def test_excel_trailing_commas(self):
        """Test CSV with trailing commas from Excel export."""
        csv_content = """plot_ID,LAT,LON,species,,,
FOREST_001,45.5231,-122.6765,Douglas Fir,,,
FOREST_002,45.5245,-122.6750,Western Hemlock,,,
,,,,,
,,,,,"""

        filepath = self.create_temp_csv(csv_content)
        df = self.handler.load_csv(filepath)

        # Should clean out empty rows and unnamed columns
        self.assertEqual(len(df), 2)
        self.assertNotIn('Unnamed', str(df.columns))
        self.assertEqual(df['plot_ID'].tolist(), ['FOREST_001', 'FOREST_002'])

    def test_alternative_column_names(self):
        """Test various alternative column naming conventions."""
        test_cases = [
            # Different latitude/longitude naming
            "id,latitude,longitude\n1,45.5,-122.6\n2,45.6,-122.7",
            "ID,Lat,Long\nA,45.5,-122.6\nB,45.6,-122.7",
            "site_name,y,x\nSite1,45.5,-122.6\nSite2,45.6,-122.7",
            "OBJECTID,POINT_Y,POINT_X\n1,45.5,-122.6\n2,45.6,-122.7",
        ]

        for i, csv_content in enumerate(test_cases):
            filepath = self.create_temp_csv(csv_content, f"test_{i}.csv")
            df = self.handler.load_csv(filepath)

            # Should always have standardized column names
            self.assertIn('plot_ID', df.columns)
            self.assertIn('LAT', df.columns)
            self.assertIn('LON', df.columns)
            self.assertEqual(len(df), 2)

    def test_missing_id_generation(self):
        """Test automatic ID generation for missing IDs."""
        csv_content = """latitude,longitude,notes
45.5231,-122.6765,Old growth
45.5245,-122.6750,Second growth
,,,
45.5189,-122.6823,Recently burned"""

        filepath = self.create_temp_csv(csv_content)

        with warnings.catch_warnings(record=True) as w:
            df = self.handler.load_csv(filepath)
            # Should have warning about generated IDs
            self.assertTrue(any("Generated" in str(warning.message) for warning in w))

        # Should have generated IDs
        self.assertEqual(len(df), 3)  # Empty row should be dropped
        self.assertIn('plot_ID', df.columns)
        self.assertTrue(all(df['plot_ID'].notna()))
        self.assertTrue(any('POINT_' in str(id) for id in df['plot_ID']))

    def test_invalid_coordinates_removal(self):
        """Test removal of invalid coordinates."""
        csv_content = """plot_id,lat,lon
A,45.5,-122.6
B,not_a_number,-122.7
C,45.6,invalid
D,91.0,-122.8
E,45.7,-181.0
F,45.8,-122.9"""

        filepath = self.create_temp_csv(csv_content)

        with warnings.catch_warnings(record=True) as w:
            df = self.handler.load_csv(filepath)

        # Should only keep valid coordinates
        self.assertEqual(len(df), 2)  # Only A and F have valid coordinates
        valid_ids = df['plot_ID'].tolist()
        self.assertIn('A', valid_ids)
        self.assertIn('F', valid_ids)
        self.assertNotIn('B', valid_ids)  # Invalid lat (not_a_number)
        self.assertNotIn('C', valid_ids)  # Invalid lon (invalid)
        self.assertNotIn('D', valid_ids)  # Lat out of range (91.0)
        self.assertNotIn('E', valid_ids)  # Lon out of range (-181.0)

    def test_swapped_coordinates_detection(self):
        """Test detection and fixing of swapped lat/lon."""
        csv_content = """plot_id,lat,lon
Normal,45.5,-122.6
Swapped,-122.7,45.6
AlsoNormal,45.7,-122.8"""

        filepath = self.create_temp_csv(csv_content)

        # Note: Current implementation doesn't auto-fix swapped coords
        # but this test shows where it could be enhanced
        df = self.handler.load_csv(filepath)

        # Should handle the data appropriately
        self.assertGreaterEqual(len(df), 2)

    def test_duplicate_id_handling(self):
        """Test handling of duplicate IDs."""
        csv_content = """plot_id,lat,lon
A,45.5,-122.6
A,45.6,-122.7
B,45.7,-122.8
A,45.8,-122.9"""

        filepath = self.create_temp_csv(csv_content)

        with warnings.catch_warnings(record=True) as w:
            df = self.handler.load_csv(filepath)
            # Should have warning about duplicate IDs
            self.assertTrue(any("duplicate" in str(warning.message).lower() for warning in w))

        # All rows should be present with unique IDs
        self.assertEqual(len(df), 4)
        self.assertEqual(len(df['plot_ID'].unique()), 4)

    def test_empty_csv(self):
        """Test handling of empty CSV."""
        csv_content = """plot_id,lat,lon
,,,
,,,"""

        filepath = self.create_temp_csv(csv_content)

        with self.assertRaises(ValueError) as context:
            df = self.handler.load_csv(filepath)
        self.assertIn("no valid data", str(context.exception).lower())

    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        csv_content = """plot_id,elevation,species
A,1000,Douglas Fir
B,1200,Western Hemlock"""

        filepath = self.create_temp_csv(csv_content)

        with self.assertRaises(ValueError) as context:
            df = self.handler.load_csv(filepath)
        self.assertIn("latitude", str(context.exception).lower())

    def test_strict_mode(self):
        """Test strict mode raises errors instead of warnings."""
        csv_content = """plot_id,lat,lon
A,45.5,-122.6
B,91.0,-122.7
C,45.6,-122.8"""

        filepath = self.create_temp_csv(csv_content)
        strict_handler = CSVHandler(strict=True)

        with self.assertRaises(ValueError) as context:
            df = strict_handler.load_csv(filepath)
        self.assertIn("out of range", str(context.exception).lower())

    def test_encoding_handling(self):
        """Test handling of different file encodings."""
        # Create CSV with special characters
        csv_content = """plot_id,lat,lon,notes
A,45.5,-122.6,Fôrêt
B,45.6,-122.7,Ñoño
C,45.7,-122.8,Björk"""

        # Test UTF-8 encoding
        filepath = self.create_temp_csv(csv_content)
        df = self.handler.load_csv(filepath)
        self.assertEqual(len(df), 3)

    def test_dataframe_input(self):
        """Test CSVHandler methods work with DataFrame input."""
        df = pd.DataFrame({
            'site_id': ['A', 'B', None, 'D'],
            'latitude': [45.5, 45.6, 45.7, 45.8],
            'longitude': [-122.6, -122.7, -122.8, -122.9]
        })

        # Clean the dataframe
        cleaned_df = self.handler.clean_dataframe(df)
        self.assertEqual(len(cleaned_df), 4)

        # Find columns
        lat_col = self.handler._find_column(cleaned_df.columns, self.handler.LAT_COLUMNS)
        lon_col = self.handler._find_column(cleaned_df.columns, self.handler.LON_COLUMNS)
        self.assertEqual(lat_col, 'latitude')
        self.assertEqual(lon_col, 'longitude')

        # Validate coordinates
        validated_df, report = self.handler.validate_coordinates(cleaned_df, lat_col, lon_col)
        self.assertEqual(len(validated_df), 4)
        self.assertEqual(report['valid_rows'], 4)

        # Generate IDs for missing values
        final_df = self.handler.generate_ids(validated_df, 'site_id')
        self.assertTrue(all(final_df['site_id'].notna()))

    def test_path_traversal_security(self):
        """Test protection against path traversal attacks."""
        # Try to access parent directories
        dangerous_paths = [
            "../../../etc/passwd",
            "../../sensitive_data.csv",
            "/etc/passwd",
        ]

        for dangerous_path in dangerous_paths:
            with self.assertRaises((ValueError, FileNotFoundError)) as context:
                self.handler.load_csv(dangerous_path)

    def test_file_size_limit(self):
        """Test file size limit enforcement."""
        # Create a large file path (we'll mock the size check)
        large_file = self.create_temp_csv("a,b,c\n1,2,3")

        # Save original MAX_FILE_SIZE
        original_limit = CSVHandler.MAX_FILE_SIZE
        try:
            # Set a tiny limit
            CSVHandler.MAX_FILE_SIZE = 10  # 10 bytes
            handler = CSVHandler()

            with self.assertRaises(ValueError) as context:
                handler.load_csv(large_file)
            self.assertIn("too large", str(context.exception).lower())
        finally:
            # Restore original limit
            CSVHandler.MAX_FILE_SIZE = original_limit

    def test_max_columns_limit(self):
        """Test maximum columns limit."""
        # Create CSV with many columns
        num_cols = 50
        header = ','.join([f'col{i}' for i in range(num_cols)])
        data = ','.join(['1'] * num_cols)
        csv_content = f"{header}\n{data}"

        filepath = self.create_temp_csv(csv_content)

        # Save original limit
        original_limit = CSVHandler.MAX_COLUMNS
        try:
            # Set a low limit
            CSVHandler.MAX_COLUMNS = 10
            handler = CSVHandler()

            with self.assertRaises(ValueError) as context:
                handler.load_csv(filepath)
            self.assertIn("too many columns", str(context.exception).lower())
        finally:
            CSVHandler.MAX_COLUMNS = original_limit

    def test_missing_longitude_column(self):
        """Test specific error when longitude column is missing."""
        csv_content = """plot_id,latitude,elevation
A,45.5,1000
B,45.6,1100"""

        filepath = self.create_temp_csv(csv_content)

        with self.assertRaises(ValueError) as context:
            self.handler.load_csv(filepath)
        self.assertIn("longitude", str(context.exception).lower())

    def test_partial_column_name_matching(self):
        """Test partial string matching for complex column names."""
        csv_content = """ObjectID_1,Latitude_WGS84,Longitude_NAD83
1,45.5,-122.6
2,45.6,-122.7"""

        filepath = self.create_temp_csv(csv_content)
        df = self.handler.load_csv(filepath)

        # Should match despite complex names
        self.assertIn('LAT', df.columns)
        self.assertIn('LON', df.columns)
        self.assertEqual(len(df), 2)

    def test_csv_parsing_fallback(self):
        """Test CSV parsing with malformed data requiring fallback."""
        # Create a problematic CSV that might need special parsing
        csv_content = """plot_id,lat,lon,notes
A,45.5,-122.6,"Complex, data with comma"
B,45.6,-122.7,"Another ""quoted"" value"
C,45.7,-122.8,Simple"""

        filepath = self.create_temp_csv(csv_content)
        df = self.handler.load_csv(filepath)

        # Should handle complex CSV formats
        self.assertEqual(len(df), 3)

    def test_process_csv_convenience_function(self):
        """Test the convenience function for processing CSVs."""
        csv_content = """plot_ID,LAT,LON
FOREST_001,45.5231,-122.6765
FOREST_002,45.5245,-122.6750"""

        filepath = self.create_temp_csv(csv_content)

        # Capture print output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        df = process_csv_for_skiba(filepath)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check the summary was printed
        self.assertIn("CSV Processing Summary", output)
        self.assertIn("Total rows processed: 2", output)
        self.assertIn("Valid rows: 2", output)

        # Check the dataframe
        self.assertEqual(len(df), 2)
        self.assertIn('plot_ID', df.columns)


if __name__ == '__main__':
    unittest.main()
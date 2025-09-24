import pandas as pd
import argparse

def load_and_display_csv(file_path: str):
    """
    Load a CSV file using pandas and display basic information.
    :param file_path: Path to the CSV file.
    """
    # Read CSV file
    df = pd.read_csv(file_path)

    # Display information
    print("===== CSV File Info =====")
    print("File path:", file_path)
    print("[Top 5 Rows]")
    print(df.head())       # Preview top 5 rows
    print("[Column Names]")
    print(df.columns)      # Show column names
    print("[Data Shape]")
    print(df.shape)        # Show (rows, columns)

    return df  # Return DataFrame if needed later

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Load and display a CSV file.')
    parser.add_argument('-i', '--input', required=True, help='Path to the CSV file')
    args = parser.parse_args()

    # Call the function
    load_and_display_csv(args.input)


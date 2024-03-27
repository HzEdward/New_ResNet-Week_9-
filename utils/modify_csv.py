# for modfiy.csv, for each row, if the [blacklisted] is 3, then set to 1
import pandas as pd
def modify_csv(csv_path):
    """
    Modify a CSV file by setting the 'blacklisted' column to 1 for rows where it is 3.

    Args:
        csv_path (str): The path to the CSV file to modify.

    """
    # Load the CSV file
    csv_path = pd.read_csv(csv_path)

    # Set 'blacklisted' to 1 for rows where it is 3
    csv_path.loc[csv_path['blacklisted'] == 3, 'blacklisted'] = 1

    

    return csv_path

if __name__ == "__main__":
    csv_path = "data_modified.csv"
    modified_csv = modify_csv(csv_path)

    # Specify the output path
    output_path = "data_modified.csv"

    # Save the modified CSV
    modified_csv.to_csv(output_path, index=False)

    print(f"Modified CSV saved to {output_path}")

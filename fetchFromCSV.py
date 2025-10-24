def fetch_line(file_path, line_number):
    try:
        with open(file_path, 'r') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == line_number:
                    print(f"Line {line_number}: {line.strip()}")
                    return line.strip()
        print(f"Line {line_number} not found in the file.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
file_path = "D:/intersection_lane_vehiclesPLow.csv"  # Replace with your file path
line_number = 404894  # Replace with the line number you want to check

fetch_line(file_path, line_number)

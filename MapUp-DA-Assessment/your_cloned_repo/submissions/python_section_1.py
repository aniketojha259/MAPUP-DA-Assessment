from typing import Dict, List

import pandas as pd
from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        temp = []
        chunk_size = min(n, length - i)
        
        # Append the reversed chunk to the temp list
        for j in range(chunk_size):
            temp.append(lst[i + chunk_size - 1 - j])
        
        # Extend the result with the reversed chunk
        result.extend(temp)
    
    """
    Reverses the input list by groups of n elements.
    """
    return result  # return the result, not the original list

n = 3
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
result1 = reverse_by_n_elements(lst, n)
print(result1)


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}  # Initialize an empty dictionary
    
    # Loop through each string in the input list
    for string in lst:
        length = len(string)  # Get the length of the string
        
        # If the length is not already a key in the dictionary, add it
        if length not in length_dict:
            length_dict[length] = []
        
        # Append the string to the list for its length
        length_dict[length].append(string)
    
    # Sort the dictionary by keys (lengths) and return it
    return dict(sorted(length_dict.items()))

import pandas as pd

def flatten_dict_pandas(nested_dict: Dict) -> Dict:
    # Convert nested dictionary to a DataFrame
    df = pd.json_normalize(nested_dict)
    return df.to_dict(orient='records')[0]

# Flatten the dictionary using pandas
result_pandas = flatten_dict_pandas(sample_dict)
print("Flattened dictionary using pandas:")
print(result_pandas)


from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    
    def backtrack(path: List[int], used: List[bool]):
        # If the path length equals the input list length, we found a valid permutation
        if len(path) == len(nums):
            result.append(path[:])  # Append a copy of the current path
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue  # Skip used elements
            
            # Skip duplicates: if current element is the same as the previous one and the previous one is not used
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            # Mark the current element as used
            used[i] = True
            path.append(nums[i])  # Add the current element to the path
            
            # Continue building the permutation
            backtrack(path, used)
            
            # Backtrack: unmark the current element and remove it from the path
            used[i] = False
            path.pop()
    
    nums.sort()  # Sort the input list to handle duplicates
    result = []  # This will hold all the unique permutations
    used = [False] * len(nums)  # Track the usage of elements
    
    backtrack([], used)  # Start the backtracking process
    
    return result

# Example usage
input_list = [1, 1, 2]
output_permutations = unique_permutations(input_list)
print(output_permutations)

def find_all_dates(text: str) -> List[str]:
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',        # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',        # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'        # yyyy.mm.dd
    ]
    
    # Combine patterns into a single regex
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    # Extract the matched dates from the groups
    valid_dates = []
    for match in matches:
        if match[0]:  # dd-mm-yyyy
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  # mm/dd/yyyy
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  # yyyy.mm.dd
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

import pandas as pd
import numpy as np
from typing import List, Tuple
import math

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decodes a Google Maps encoded polyline into a list of (lat, lon) tuples."""
    index = 0
    lat = 0
    lon = 0
    coordinates = []
    
    while index < len(polyline_str):
        # Decode latitude
        b = 0
        shift = 0
        while True:
            value = ord(polyline_str[index]) - 63
            index += 1
            b |= (value & 0x1f) << shift
            shift += 5
            if value < 0x20:
                break
        lat += ~(b >> 1) if (b & 1) else (b >> 1)
        
        # Decode longitude
        b = 0
        shift = 0
        while True:
            value = ord(polyline_str[index]) - 63
            index += 1
            b |= (value & 0x1f) << shift
            shift += 5
            if value < 0x20:
                break
        lon += ~(b >> 1) if (b & 1) else (b >> 1)
        
        # Append the decoded point
        coordinates.append((lat / 1E5, lon / 1E5))
    
    return coordinates

def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculates the distance in meters between two points on the Earth."""
    R = 6371000  # Earth radius in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into latitude and longitude coordinates
    coordinates = decode_polyline(polyline_str)
    
    # Create lists to hold latitude, longitude, and distances
    latitudes = []
    longitudes = []
    distances = [0.0]  # Start with 0 distance for the first point
    
    # Iterate over coordinates to fill latitudes and longitudes
    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:
            distance = haversine(coordinates[i - 1], (lat, lon))
            distances.append(distance)
    
    # Create a DataFrame from the data
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance_m': distances
    })
    
    return df

polyline_str = "u{~vF~gu~aP"
df = polyline_to_dataframe(polyline_str)
print(df)


from typing import List

def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """Rotate the n x n matrix by 90 degrees clockwise."""
    n = len(matrix)
    # Create a new matrix for the rotated version
    rotated = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    return rotated

def transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """Transform the matrix by replacing each element with the sum of its row and column, excluding itself."""
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row and current column excluding the current element
            row_sum = sum(matrix[i]) - matrix[i][j]
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
            transformed[i][j] = row_sum + col_sum
            
    return transformed

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in the same row and column (in the rotated matrix), excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    rotated_matrix = rotate_matrix(matrix)  # Rotate the matrix by 90 degrees
    final_matrix = transform_matrix(rotated_matrix)  # Transform the rotated matrix
    return final_matrix

# Example usage
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

result = rotate_and_multiply_matrix(matrix)
print(result)



import numpy as np

# Mapping days to numerical values
day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

def time_check(df) -> pd.Series:
    """
    Checks if the timestamps for each unique (id, id_2) pair cover a full 24-hour period
    and span all 7 days of the week.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the columns id, id_2, startDay, startTime, endDay, endTime.
    
    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps, indexed by (id, id_2).
    """
    # Convert startDay and endDay to numerical values using the day_map
    df['startDay_num'] = df['startDay'].map(day_map)
    df['endDay_num'] = df['endDay'].map(day_map)

    # Initialize a set for all days (0 to 6) and all hours (0 to 23)
    all_days = set(range(7))
    all_hours = set(range(24))

    # Group the DataFrame by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Initialize a dictionary to track if each (id, id_2) pair covers all days and hours
    completeness_check = {}

    for (group_id, group_id2), group in grouped:
        # Track the days and hours covered by the current group
        covered_days = set()
        covered_hours = set()
        
        for _, row in group.iterrows():
            # Get start and end days
            start_day = row['startDay_num']
            end_day = row['endDay_num']

            # Handle the time range within a day
            start_hour = pd.to_datetime(row['startTime']).hour
            end_hour = pd.to_datetime(row['endTime']).hour

            # Calculate the range of days covered
            if start_day <= end_day:
                days_range = range(start_day, end_day + 1)
            else:
                # Wrap around the week (Sunday to Monday)
                days_range = list(range(start_day, 7)) + list(range(0, end_day + 1))
            
            # Update covered days
            covered_days.update(days_range)

            # Update covered hours
            if start_hour <= end_hour:
                hours_range = range(start_hour, end_hour + 1)
            else:
                # Wrap around the day (23:59 to 00:00)
                hours_range = list(range(start_hour, 24)) + list(range(0, end_hour + 1))
            covered_hours.update(hours_range)

        # Check if all days and hours are covered
        days_complete = all_days.issubset(covered_days)
        hours_complete = all_hours.issubset(covered_hours)
        completeness_check[(group_id, group_id2)] = not (days_complete and hours_complete)

    # Create a boolean Series with MultiIndex from the completeness_check dictionary
    result = pd.Series(completeness_check)
    result.index = pd.MultiIndex.from_tuples(result.index, names=['id', 'id_2'])
    
    return result

# Apply the function to the dataset
incomplete_timestamps = time_check(df)

# Display the results
incomplete_timestamps.head()


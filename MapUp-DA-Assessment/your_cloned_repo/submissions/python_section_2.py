import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Get all unique toll locations
    locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Create a distance matrix initialized with infinity (to represent unknown distances)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    
    # Set the diagonal to zero (distance from a location to itself)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Update the matrix with the known distances from the dataset
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.loc[start, end] = dist
        distance_matrix.loc[end, start] = dist  # Symmetric update
    
    # Apply the Floyd-Warshall algorithm to find the shortest paths
    for k in locations:
        for i in locations:
            for j in locations:
                # Update the distance matrix to be the minimum distance through intermediate nodes
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], 
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    return distance_matrix

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Display the resulting distance matrix
distance_matrix.head()



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create a set of unique ids from both id_start and id_end
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel())

    # Generate all possible combinations of id_start and id_end (excluding cases where id_start == id_end)
    unrolled_data = [(i, j, df[(df['id_start'] == i) & (df['id_end'] == j)]['distance'].values[0])
                     for i in ids for j in ids if i != j and len(df[(df['id_start'] == i) & (df['id_end'] == j)]) > 0]
    
    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df

# Applying the function to the dataset
unrolled_df = unroll_distance_matrix(df)
unrolled_df.head()



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.
    
    Args:
        df (pandas.DataFrame)
        reference_id (int)
    
    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference_id
    reference_avg = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the 10% threshold (upper and lower bounds)
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

    # Find all other IDs whose average distance falls within this range
    result_ids = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = result_ids[(result_ids >= lower_bound) & (result_ids <= upper_bound)].index.tolist()

    # Sort the IDs
    ids_within_threshold.sort()

    return ids_within_threshold

# Example usage with a sample reference_id (replace with actual reference_id from the dataset)
reference_id = 1001402
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
ids_within_threshold



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pandas.DataFrame
    """
    # Calculate the toll rates for each vehicle type based on distance and the given rate coefficients
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6
    
    return df

# Applying the function to the unrolled dataframe
toll_rates_df = calculate_toll_rate(unrolled_df)
toll_rates_df.head()



import datetime

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define the days of the week and time intervals
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)
    ]

    # Prepare an empty list to store the expanded rows
    expanded_rows = []

    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        for day in days_of_week:
            # Determine if it's a weekend or a weekday
            if day in ['Saturday', 'Sunday']:
                discount_factor = 0.7
                # For weekends, the time range is constant for all hours of the day
                expanded_rows.append({
                    **row.to_dict(),
                    'start_day': day,
                    'end_day': day,
                    'start_time': datetime.time(0, 0, 0),
                    'end_time': datetime.time(23, 59, 59),
                    'moto': row['moto'] * discount_factor,
                    'car': row['car'] * discount_factor,
                    'rv': row['rv'] * discount_factor,
                    'bus': row['bus'] * discount_factor,
                    'truck': row['truck'] * discount_factor
                })
            else:
                # For weekdays, iterate over the three time intervals
                for start_time, end_time, discount_factor in time_intervals:
                    expanded_rows.append({
                        **row.to_dict(),
                        'start_day': day,
                        'end_day': day,
                        'start_time': start_time,
                        'end_time': end_time,
                        'moto': row['moto'] * discount_factor,
                        'car': row['car'] * discount_factor,
                        'rv': row['rv'] * discount_factor,
                        'bus': row['bus'] * discount_factor,
                        'truck': row['truck'] * discount_factor
                    })

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

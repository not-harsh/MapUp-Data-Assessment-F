
import pandas as pd
import datetime
import pandas as pd
import networkx as nx

#provide appropriate file paths
dataset1filepath = 'D:\Mapup\datasets\dataset-1.csv'
dataset2filepath = 'D:\Mapup\datasets\dataset-2.csv'
dataset3filepath = 'D:\Mapup\datasets\dataset-3.csv'

#Question 1: Distance Matrix Calculation
def calculate_distance_matrix(df)->pd.DataFrame():

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], weight=row['distance'])  # Add bidirectional edge

    ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=ids, columns=ids)

    # Populate 
    for start_id in ids:
        for end_id in ids:
            if start_id == end_id:
                distance_matrix.loc[start_id, end_id] = 0.0
            elif nx.has_path(G, start_id, end_id):
                distance_matrix.loc[start_id, end_id] = nx.shortest_path_length(G, start_id, end_id, weight='weight')
            else:
                distance_matrix.loc[start_id, end_id] = float('inf')

    return distance_matrix

df = pd.read_csv(dataset3filepath)
result_df = calculate_distance_matrix(df)
print("DISTANCE MATRIX CALCULATION:\n")
print(result_df,"\n\n\n")


#Question 2: Unroll Distance Matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    for id_start, row in df.iterrows():
        for id_end, distance in row.items():
            if id_start != id_end:
                unrolled_df = pd.concat([unrolled_df, pd.DataFrame([[id_start, id_end, distance]], columns=['id_start', 'id_end', 'distance'])], ignore_index=True)

    return unrolled_df

unrolled_result_df = unroll_distance_matrix(result_df)
print("UNROLL DISTANCE MATRIX:\n")
print(unrolled_result_df,"\n\n\n")


#Question 3: Finding IDs within Percentage Threshold

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    reference_rows = df[df['id_start'] == reference_id]

    # avg distance for refrence value
    reference_avg_distance = reference_rows['distance'].mean()

    # calculate the threshold range 
    threshold = 0.1 * reference_avg_distance

    within_threshold_ids = df[(df['distance'] >= reference_avg_distance - threshold) & (df['distance'] <= reference_avg_distance + threshold)]['id_start'].unique()

    # sort
    sorted_within_threshold_ids = sorted(within_threshold_ids)

    return sorted_within_threshold_ids

reference_id = 1001400  # Replace with the desired reference value
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_result_df, reference_id)
print("IDs WITHIN PERCENTAGE THRESHOLD:\n")
print(ids_within_threshold,"\n\n\n")


#Question 4: Calculate Toll Rate

def calculate_toll_rate(df)->pd.DataFrame():
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

toll_rate_result_df = calculate_toll_rate(unrolled_result_df)
print("CALCULATE TOLL RATE:\n")
print(toll_rate_result_df,"\n\n\n")


# Question 5: Calculate Time-Based Toll Rates

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    
    time_ranges_weekdays = [(datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
                            (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
                            (datetime.time(18, 0, 0), datetime.time(23, 59, 59))]
    
    time_ranges_weekends = [(datetime.time(0, 0, 0), datetime.time(23, 59, 59))]
    
    discount_factors_weekdays = [0.8, 1.2, 0.8]
    discount_factor_weekends = 0.7
    
    result_dfs = []

    for _, group in df.groupby(['id_start', 'id_end']):
        id_start, id_end = _
        distance = group['distance'].iloc[0] 
        
        for day in range(7):
            start_day = (datetime.datetime(2023, 1, 1) + datetime.timedelta(days=day)).strftime('%A')
            
            for time_range in (time_ranges_weekdays if day < 5 else time_ranges_weekends):
                start_time, end_time = time_range
                discount_factor = discount_factors_weekdays[time_ranges_weekdays.index(time_range)] if day < 5 else discount_factor_weekends
                
                # calculating tolls
                toll_rates = {vehicle: group[vehicle] * discount_factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                
                # appending results
                result_dfs.append(pd.DataFrame({
                    'id_start': [id_start],
                    'id_end': [id_end],
                    'distance': [distance],
                    'start_day': [start_day],
                    'start_time': [start_time],
                    'end_day': [start_day],
                    'end_time': [end_time],
                    **toll_rates
                }))
    
    # Concat
    result_df = pd.concat(result_dfs, ignore_index=True)
    
    return result_df

result_df = calculate_time_based_toll_rates(unrolled_result_df)
print("CALCULATE TIME-BASED TOLL RATES:\n")
print(result_df)




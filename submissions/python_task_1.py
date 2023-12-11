import pandas as pd

#provide appropriate file paths
dataset1filepath = 'D:\Mapup\datasets\dataset-1.csv'
dataset2filepath = 'D:\Mapup\datasets\dataset-2.csv'
dataset3filepath = 'D:\Mapup\datasets\dataset-3.csv'


# Question 1: Car Matrix Generation

#using loop
def generate_car_matrix(df)->pd.DataFrame:
    # Loading dataset
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """

    # Create a pivot table with id_1 as index, id_2 as columns, and car as values
    car_matrix = df.pivot_table(index='id_1', columns='id_2', values='car', fill_value=0)

    # Setting diagonal values to zeroes
    for i in car_matrix.index:
        car_matrix.at[i, i] = 0

    return car_matrix
df = pd.read_csv(dataset1filepath)
result_matrix = generate_car_matrix(df)
print("CAR MATRIX GENERATION:\n")
print(result_matrix,"\n\n\n")


# Question 2: Car Type Count Calculation
def get_type_count(df) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'car' column.

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Defining car type categories
    bins = [-float('inf'), 15, 25, float('inf')]
    labels = ['low', 'medium', 'high']

    # creating 'car_type' based on values of 'car' columns
    df['car_type'] = pd.cut(df['car'], bins=bins, labels=labels, right=False)

    # counting car type and sorting
    type_count = df['car_type'].value_counts().sort_index().to_dict()

    return type_count


df = pd.read_csv(dataset1filepath)
result = get_type_count(df)
print("CAR TYPE COUNT CALCULATION:\n")
print(result,"\n\n\n")


# Question 3: Bus Count Index Retrieval

def get_bus_indexes(df) -> list:
    """
    Identify and return the indices where bus values are greater than twice the mean value.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'bus' column.

    Returns:
        list: A list of indices (sorted in ascending order) where bus values are greater than twice the mean.
    """
    # mean of bus column
    bus_mean = df['bus'].mean()

    # identifying indices where bus value is greater than twice the mean value of bus column
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # sorting
    bus_indexes.sort()

    return bus_indexes

df = pd.read_csv(dataset1filepath)
result = get_bus_indexes(df)
print("BUS COUNT INDEX RETRIEVAL:\n")
print(result,"\n\n\n")


# Question 4: Route Filtering
def filter_routes(df) -> list:
    """
    Return the sorted list of values of column 'route' for which the average of 'truck' values is greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'route' and 'truck' columns.

    Returns:
        list: A sorted list of values of 'route' column.
    """
    df = pd.read_csv(dataset1filepath)
    # calculating average of truck
    route_avg_truck = df.groupby('route')['truck'].mean()

    # average greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # sorting
    filtered_routes.sort()

    return filtered_routes


df = pd.read_csv(dataset1filepath)
result = filter_routes(df)
print("ROUTE FILTERING:\n")
print(result,"\n\n\n")


# Question 5: Matrix Value Modification
def multiply_matrix(df) -> pd.DataFrame:
    """
    Modify each value in the DataFrame based on specified logic.

    If a value is greater than 20, multiply by 0.75.
    If a value is 20 or less, multiply by 1.25.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame with values rounded to 1 decimal place.
    """
    
    modified_df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # round values
    modified_df = modified_df.round(1)

    return modified_df


result_matrix_modified = multiply_matrix(result_matrix)
print("MATRIX VALUE MODIFICATION:\n")
print(result_matrix_modified,"\n\n\n")


# Question 6: Time Check
import warnings
def check_timestamp_completeness(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #warnings.simplefilter('ignore', category=FutureWarning)
        df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%I:%M:%S %p', errors='coerce')

        df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%I:%M:%S %p', errors='coerce')

        df.set_index(['id', 'id_2'], inplace=True)

        result_series = pd.Series(index=df.index, dtype=bool)

        for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):

            full_day_coverage = (group['end_timestamp'].max() - group['start_timestamp'].min()) >= pd.Timedelta(days=1)

            days_of_week_coverage = set(group['start_timestamp'].dt.day_name()).issuperset(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

            result_series.loc[(id_val, id_2_val)] = not (full_day_coverage and days_of_week_coverage)

        result_df = pd.DataFrame(result_series, columns=['result'])

    return result_df

df = pd.read_csv(dataset2filepath)
result = check_timestamp_completeness(df)
print("TIME CHECK:\n")
print(result)
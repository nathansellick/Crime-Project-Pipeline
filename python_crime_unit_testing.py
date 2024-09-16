import pandas as pd
import glob
import os 
import pytest 
import logging
import tempfile
from unittest import mock

# Defining functions
# Defining function for concatenating multiple csv's in a folder
def concat_csvs(file_path: str):
    """
    Pass in file path of folder containing crime csv's and concatenate to return one dataframe 
    """
    logging.info(f"Starting csv concatenation from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"Folder not found: {file_path}")
        raise FileNotFoundError(f"Folder not found: {file_path}")
    try:
        csv_files = glob.glob(os.path.join(file_path, '**', '*.csv'), recursive=True)
        dataframes = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"CSV concatenation from {file_path} completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error reading the CSV files in {file_path}: {e}")
        raise ValueError(f"Error reading the CSV files in {file_path}: {e}")

# Defining function for ingesting data 
def ingest_data(file_path: str)-> pd.DataFrame:
    """
    Ingest raw data from a CSV file. Pass in the file path as a string and returns a pandas dataframe.
    """
    logging.info(f"Starting data ingestion from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data ingestion from {file_path} completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error reading the CSV file {file_path}: {e}")
        raise ValueError(f"Error reading the CSV file {file_path}: {e}")
    

# Defining function for deleting columns 
def del_cols(df: pd.DataFrame, list_of_cols:list): #: followed by df/ list indicates expected input 
    """
    Delete unnecessary column(s) specified in list_of_cols from the DataFrame.
    """
    df = df.drop(columns=list_of_cols)
    return df

# Defining function for renaming columns 
def rename_cols(df: pd.DataFrame, dict_of_names:dict):
    """
    Rename column(s) specified in list_of_cols from the DataFrame
    """
    df.rename(columns = dict_of_names,inplace = True)
    return df

# Defining function for splitting reported by column
def split_reported_by(df):
    """
    Separate the column 'Reported by' and create a new column 'County' that just gives county name e.g. 'Surrey Police' to 'Surrey' 
    """
    df['County'] = df['Reported by'].str.split(' ',n=1).str[0] # n=1 indicates split at first delimiter (space in this case) and then .str[0] the first element
    return df

# Defining function for mapping nulls in Crime ID
def map_null_crime_id(df):
    """
    Map the null values in 'Crime ID' column to say 'No Crime ID' 
    """ 
    df['Crime ID'] = df['Crime ID'].fillna('No Crime ID')

    return df

def rstrip_columns(df):
    """
    Remove trailing characters from the end of any rows in columns of datatype object
    """
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column is of type string
            df[column] = df[column].str.strip()
    return df

def round_to_3dp(df):
    """
    Rounds longitude and latitude floats in a dataframe to all be only 3 decimal places long
    """
    df['Longitude'] = df['Longitude'].round(3)
    df['Latitude'] = df['Latitude'].round(3)
    return df

def drop_null_rows(df: pd.DataFrame, list_of_cols: list):
    """
    Deletes any rows from a dataframe where there is a null in any of column(s) given in list_of_cols
    """
    df.dropna(subset = list_of_cols, inplace=True)
    return df

def drop_dup_rows(df: pd.DataFrame, list_of_cols: list):
    """
    Deletes any rows that are duplicates in column(s) specified in list_of_cols
    """
    df = df.drop_duplicates(subset= list_of_cols, keep='first')
    return df

def left_merge(df1: pd.DataFrame, df2: pd.DataFrame, list_of_cols: list):
    """
    left join of two dataframes on column(s) specified in list_of_cols
    """
    return pd.merge(df1, df2, on = list_of_cols, how='left')


# Defining function for categorising last outcome category
def categorise_outcome(outcome):
    """
    categorise each outcome to give the final outcome category
    """
    if outcome in ['Unable to prosecute suspect', 'Investigation complete; no suspect identified', 'Status update unavailable']: 
        return 'No Further Action'
    elif outcome in ['Local resolution', 'Offender given a caution', 'Action to be taken by another organisation']:
        return 'Non-criminal Outcome'
    elif outcome in ['Further investigation is not in the public interest', 'Further action is not in the public interest', 'Formal action is not in the public interest']: #
        return 'Public Interest Consideration'
    else:
        return 'Unknown'  
    
# Defining function to apply categorise_outcome to last outcome category and create new column

def apply_outcome_categorisation(df: pd.DataFrame):
    """
    Apply categorisation to 'Outcome' and put result in new column 'Final outcome category'
    """
    df['Final outcome'] = df['Last outcome category'].apply(categorise_outcome)

    return df

# Defining function for categorising crime type 
def categorise_crime_type(crime_type):
    """
    categorise each crime type into 1 of 6 categories
    """
    if crime_type in ['Burglary', 'Robbery']: 
        return 'Burglary + robbery'
    elif crime_type in ['Other theft','Vehicle crime','Shoplifting','Bicycle theft','Theft from the person']:
        return 'Theft-related crimes'
    elif crime_type in ['Violence and sexual offences','Possession of weapons','Drugs']: #
        return 'Violent/ serious offences'
    elif crime_type in ['Anti-social behaviour','Public order']:
        return 'Anti-social behaviour/ disorder'
    elif crime_type in ['Criminal damage and arson']:
        return 'Property damage'
    elif crime_type in ['Other crime']:
        return 'Miscellaneous/ other crimes'
    else:
        return 'Unknown'
    
# Defining function for applying categorise_crime_type to crime type and creating new column
def apply_crime_type_categorisation(df):
    """
    Apply categorisation to 'Crime type' and put result in new column 'Crime type category'
    """
    df['Crime type category'] = df['Crime type'].apply(categorise_crime_type)

    return df

# Defining function for concatting property csv's
def concat_property_csvs(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    concatenate two dataframes
    """
    return pd.concat([df1,df2],ignore_index=True)

# Defining function for grouping by county 
def group_by(df: pd.DataFrame, list_of_cols: list, agg_dict: dict):
    """
    group a dataframe by list of columns specified in list_of_cols and aggregate by columns and aggregation types in agg_dict
    """
    return df.groupby(list_of_cols).agg(agg_dict)

# Defining function for filtering to get only burglary + robbery 

def crime_category_filter(df:pd.DataFrame, filter_list: list):
    """
    filter dataframe to show only crime type categories that are in filter_list
    """
    return df[df['Crime type category'].isin(filter_list)]

# Defining function for creating new column with burglaries + robberies per km^2
def burg_rob_km2(df):
    """
    creating a burglaries + robberies per km^2 column
    """
    df['Burglaries + Robberies per km\u00B2'] = (df['Crime type category']/df['Area (km2)']).round(2)
    return df

def split_postcode(df):
    """
    Separate the postcode column to obtain just the first 3/4 characters as 'Postcode start'
    """
    df['Postcode start'] = df['Postcode'].str.split(' ',n=1).str[0] # n=1 indicates split at first delimiter (space in this case) and then .str[0] the first element
    return df

# Defining function for filtering by county
def county_filter(df: pd.DataFrame, list_filter: list):
    """
    Function to filter a dataframe and only look at data for specific counties in list_filter
    """
    return df[df['County'].isin(list_filter)]

def sort_dataframe(df: pd.DataFrame, col: str, asc: bool):
    """
    sort a dataframe by a column (col) and specify whether ascending or not as boolean
    """
    return df.sort_values(by = col, ascending = asc )

def df_reset_index(df: pd.DataFrame):
    """
    moves current index inside dataframe
    """
    df.reset_index(drop = False, inplace = True)
    return df


# Test all functions given so far 

#First testing concat_csvs and ingest_data functions by Mock tests 

# Mocking os.path.exists and glob.glob for file-related functions
@mock.patch('os.path.exists', return_value=True) #forcing that directory exists with True
@mock.patch('glob.glob', return_value=['file1.csv', 'file2.csv']) #creates two mock csv's in directory
@mock.patch('pandas.read_csv') #replaces pd.read_csv with a mock function
def test_concat_csvs_valid(mock_read_csv, mock_glob, mock_exists): #mocked versions above are inputted as arguments
    """
    several tests that concat_csvs function works with valid directory and mock csv's. Test include checking dimensions of dataframe/ number of calls etc.
    """
    # Simulate reading the CSV files
    mock_read_csv.side_effect = [
        pd.DataFrame({'A': [1, 2]}), # side_effect allows different DataFrames to be returned for each file 
        pd.DataFrame({'A': [3, 4]})
    ]
    
    result_df = concat_csvs("valid_path")
    
    # Assert the shape of the concatenated DataFrame is correct
    assert result_df.shape == (4, 1)
    
    # Ensure os.path.exists was called correctly
    mock_exists.assert_called_once_with("valid_path")
    
    # Ensure glob.glob was called to search for CSV files
    mock_glob.assert_called_once_with(os.path.join("valid_path", '**', '*.csv'), recursive=True)
    
    # Ensure pandas.read_csv was called twice (once for each CSV)
    assert mock_read_csv.call_count == 2

# Test case where the directory does not exist
@mock.patch('os.path.exists', return_value=False)
def test_concat_csvs_invalid_path(mock_exists):
    """
    tests that FileNotFoundError is returned witn an invalid file path 
    """
    with pytest.raises(FileNotFoundError):
        concat_csvs("invalid_path")
    # Ensure os.path.exists was called correctly
    mock_exists.assert_called_once_with("invalid_path")

import pytest
from unittest import mock

# Mocking os.path.exists and pandas.read_csv for file-related functions
@mock.patch('os.path.exists', return_value=True) # file exists
@mock.patch('pandas.read_csv') # mock version of pd.read_csv function
def test_ingest_data_valid(mock_read_csv, mock_exists):
    # Simulate a valid DataFrame being returned by read_csv
    mock_read_csv.return_value = pd.DataFrame({'A': [1, 2]})
    
    result_df = ingest_data("valid_file.csv")
    
    # Assert the DataFrame is read correctly
    assert result_df.equals(pd.DataFrame({'A': [1, 2]}))
    
    # Ensure os.path.exists was called correctly
    mock_exists.assert_called_once_with("valid_file.csv")
    
    # Ensure pandas.read_csv was called once
    mock_read_csv.assert_called_once_with("valid_file.csv")

# Test case where the file does not exist
@mock.patch('os.path.exists', return_value=False)
def test_ingest_data_file_not_found(mock_exists):
    with pytest.raises(FileNotFoundError):
        ingest_data("nonexistent_file.csv")
    # Ensure os.path.exists was called correctly
    mock_exists.assert_called_once_with("nonexistent_file.csv")



# Defining fixture which returns sample dataframe which will be called as argument for several tests

@pytest.fixture
def sample_df():
    data = {
    'Crime ID': [None,'4agy'],
    'Month': ['2022-01','2022-02'],
    'Reported by': ['Surrey Police','Essex Police'],
    'Last outcome category': ['Unable to prosecute suspect',None],
    'Crime type': ['Burglary','Shoplifting'],
    'Longitude': [3.4672,3.4672],
    'Latitude': [8.7889,8.7889]
    }
    df_test = pd.DataFrame(data)
    return df_test


# Test del_cols function 
def test_del_cols_valid(sample_df):
    """
    Tests del_cols function by checking 'Crime ID' is not a column label in new dataframe
    """
    sample_df = del_cols(sample_df,['Crime ID'])
    assert 'Crime ID' not in sample_df.columns



# Test rename_cols function
def test_rename_cols_valid(sample_df):
    """
    Tests rename_cols function by checking 'Crime ID' is renamed to 'ID'
    """
    rename_cols(sample_df,{'Crime ID':'ID'})
    assert 'ID' in sample_df.columns

# Test split_reported_by function
def test_split_reported_by_valid(sample_df):
    """
    Tests split_reported_by function by checking 'Surrey Police' is split into two columns containing 'Surrey' and 'Police'
    """
    split_reported_by(sample_df)
    assert sample_df['County'][0] == 'Surrey'

# Test map_null_crime_id
def test_map_null_crime_id_valid(sample_df):
    """
    Tests map_null_crime_id function by checking null value in dataframe is changed to 'No Crime ID'
    """
    map_null_crime_id(sample_df)
    assert sample_df['Crime ID'][0] == 'No Crime ID'




# Test round_to_3dp
def test_round_to_3dp_valid(sample_df):
    """
    Tests round_to_3dp function by checking two rows have been rounded to desired floats 
    """
    round_to_3dp(sample_df)
    assert sample_df['Longitude'][0] == 3.467 and sample_df['Latitude'][1] == 8.789

# Test drop_null_rows
def test_drop_null_rows_valid(sample_df):
    """
    Tests drop_null_rows function by checking one row of null data has been deleted from dataframe
    """
    drop_null_rows(sample_df,['Crime ID','Month'])
    assert len(sample_df) == 1

# Test drop_dup_rows
def test_drop_dup_rows_valid(sample_df):
    """
    Tests drop_duplicate_rows function by checking one row of duplicate data in columns specified has been deleted 
    """
    round_to_3dp(sample_df)
    sample_df = drop_dup_rows(sample_df,['Longitude','Latitude'])
    assert len(sample_df) == 1
   


def test_categorise_outcome_valid():
    """
    tests categorise_outcome function by checking an example list of last outcomes type are categorised to a final outcome
    """
    # Example list of outcomes to test
    test_list = ['Unable to prosecute suspect','Local resolution','Further investigation is not in the public interest',None]

    # Expected outcomes based on the input test_list
    expected_list = ['No Further Action', 'Non-criminal Outcome', 'Public Interest Consideration', 'Unknown']

    # Apply categorise_outcome function to each item in the test_list
    categorised_list = [categorise_outcome(i) for i in test_list]

    # Assert the output matches the expected list
    assert categorised_list == expected_list

def test_apply_outcome_categorisation_valid(sample_df):
    """
    tests apply_categorisation by checking expected final outcome is in the dataframe
    """
    apply_outcome_categorisation(sample_df)
    assert sample_df['Final outcome'][0] == 'No Further Action'

def test_categorise_crime_type_valid():
    """
    tests categorise_crime_type function by checking an example list of last crime types are categorised to correct crime category
    """
    test_list = ['Burglary','Other theft','Violence and sexual offences','Anti-social behaviour','Criminal damage and arson','Other crime',None]

    expected_list = ['Burglary + robbery','Theft-related crimes','Violent/ serious offences','Anti-social behaviour/ disorder','Property damage','Miscellaneous/ other crimes','Unknown']

    categorised_list = [categorise_crime_type(i) for i in test_list]

    assert categorised_list == expected_list

def test_apply_crime_type_categorisation_valid(sample_df):
    """
    Tests apply crime_type_categorisation function by checking expected crime type category 'Burglary + robbery' is in dataframe
    """
    apply_crime_type_categorisation(sample_df)
    assert sample_df['Crime type category'][0] == 'Burglary + robbery'


def test_concat_property_csvs_valid():
    """
    Test concat_property_csvs with valid DataFrames.
    """
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result_df = concat_property_csvs(df1, df2)
    assert result_df.shape == (4, 1)  # There should be 4 rows

def test_concat_property_csvs_invalid():
    """
    Test concat_property_csvs with one valid and one invalid DataFrame.
    """
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = [10,20]
    with pytest.raises(TypeError):
        concat_property_csvs(df1, df2)

def test_split_postcode_valid():
    """
    Test split_postcode with valid data.
    """
    df = pd.DataFrame({'Postcode': ['AB12 3CD', 'EF45 6GH']})
    result_df = split_postcode(df)
    expected = pd.DataFrame({'Postcode': ['AB12 3CD', 'EF45 6GH'], 'Postcode start': ['AB12', 'EF45']})
    pd.testing.assert_frame_equal(result_df, expected)

def test_split_postcode_invalid():
    """
    Test split_postcode with dataframe that has column title of 'Postcode' mispelt as 'Postcod'
    """
    df = pd.DataFrame({'Postcod': [None, 'EF45 6GH']})
    with pytest.raises(KeyError):
        split_postcode(df)

def test_county_filter_valid():
    """
    Test county_filter with valid counties.
    """
    df = pd.DataFrame({'County': ['Surrey', 'Essex']})
    result_df = county_filter(df, ['Surrey'])
    assert len(result_df) == 1  # Only one matching row

def test_county_filter_invalid():
    """
    Test county_filter with an invalid county name.
    """
    df = pd.DataFrame({'County': ['Surrey', 'Essex']})
    result_df = county_filter(df, ['Non-existent County'])
    assert len(result_df) == 0  # No matching rows











import pandas as pd
import glob
import os 
import pytest 
import logging

# Constants
LOCAL_DATA_PATH = './'
LOG_FILE = os.path.join(LOCAL_DATA_PATH, 'pipeline.log')
RAW_ROOT_DIRECTORY = os.path.join(LOCAL_DATA_PATH, 'Police crime dataset')
RAW_ESSEX_PROPERTY = os.path.join(LOCAL_DATA_PATH,'essex property dataset.csv')
RAW_KENT_PROPERTY = os.path.join(LOCAL_DATA_PATH,'kent property dataset.csv')
RAW_POSTCODE = os.path.join(LOCAL_DATA_PATH,'ukpostcodes.csv')
STAGED_CRIME = os.path.join(LOCAL_DATA_PATH, 'staged_crime.csv')
STAGED_POSTCODE = os.path.join(LOCAL_DATA_PATH, 'staged_postcode.csv')
STAGED_KENT_PROPERTY = os.path.join(LOCAL_DATA_PATH, 'staged_kent_property.csv')
STAGED_ESSEX_PROPERTY = os.path.join(LOCAL_DATA_PATH, 'staged_essex_property.csv')
PRIMARY_CRIME = os.path.join(LOCAL_DATA_PATH, 'primary crime.csv')
PRIMARY_PROPERTY = os.path.join(LOCAL_DATA_PATH, 'primary property.csv')
PRIMARY_COUNTY_AREA = os.path.join(LOCAL_DATA_PATH, 'county areas.csv')
REPORTING_BURGLARY_ROBBERY_DENSITY = os.path.join(LOCAL_DATA_PATH, 'reporting county burglary + robbery densities.csv')
REPORTING_ESSEX_MONTHLY_DATA = os.path.join(LOCAL_DATA_PATH, 'reporting essex monthly data.csv')
REPORTING_KENT_MONTHLY_DATA = os.path.join(LOCAL_DATA_PATH, 'reporting kent monthly data.csv')
REPORTING_POSTCODE_CRIME = os.path.join(LOCAL_DATA_PATH,'reporting postcode crime.csv')


# Configure logging

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

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



#staging layer 
def staging():
    """
    Ingest the raw data, apply cleaning and store to CSV files for staging
    """
    logging.info("Starting Staging Layer")
     # ingest raw data
    crime_df = concat_csvs(RAW_ROOT_DIRECTORY)
    postcode_df = ingest_data(RAW_POSTCODE)
    kent_df = ingest_data(RAW_KENT_PROPERTY)
    essex_df = ingest_data(RAW_ESSEX_PROPERTY)
    try:

        # Create column that gives county name for each crime 
        crime_df = split_reported_by(crime_df)

        # Deleting unncessary columns 
        crime_df = del_cols(crime_df,['Reported by','Falls within', 'Context','LSOA name','LSOA code'])
        essex_df = del_cols(essex_df,['URI','Region GSS code','Reporting period','Pivotable date'])
        kent_df = del_cols(kent_df,['URI','Region GSS code','Reporting period','Pivotable date'])
        postcode_df = del_cols(postcode_df,['id'])

        # Renaming columns
        essex_df = rename_cols(essex_df,{'Name':'County', 'Period':'Month'})
        kent_df = rename_cols(kent_df,{'Name':'County', 'Period':'Month'})
        postcode_df = rename_cols(postcode_df,{'postcode':'Postcode','longitude':'Longitude','latitude':'Latitude'})
        


        # Dealing with null crime ID's  
        crime_df = map_null_crime_id(crime_df)

        # Remove trailing characters
        crime_df = rstrip_columns(crime_df)
        essex_df = rstrip_columns(essex_df)
        kent_df = rstrip_columns(kent_df)
        postcode_df = rstrip_columns(postcode_df)

        # Save staging files to csv
        crime_df.to_csv(STAGED_CRIME, index = False)
        essex_df.to_csv(STAGED_ESSEX_PROPERTY, index = False)
        kent_df.to_csv(STAGED_KENT_PROPERTY, index = False)
        postcode_df.to_csv(STAGED_POSTCODE, index = False)
       

        logging.info("Staging layer completed successfully")

    except Exception as e:
        logging.error(f"Error during Staging Layer: {e}")

# Primary layer

def primary():
    """
    Primary layer: Store the transformed data to CSV files
    """
    logging.info("Starting Primary Layer")
    # ingest staging
    crime_df = ingest_data(STAGED_CRIME)
    essex_df = ingest_data(STAGED_ESSEX_PROPERTY)
    kent_df = ingest_data(STAGED_KENT_PROPERTY)
    postcode_df = ingest_data(STAGED_POSTCODE)
    try:

        # Round longitude and latitude to 3 decimal places
        crime_df = round_to_3dp(crime_df)
        postcode_df = round_to_3dp(postcode_df)

        # Modify postcode dataset in preparation of merging
        postcode_df = drop_null_rows(postcode_df,['Longitude','Latitude'])
        postcode_df = drop_dup_rows(postcode_df,['Longitude','Latitude'])

        # Merge staged crime and postcode dataframes
        mdf = left_merge(crime_df, postcode_df, ['Longitude','Latitude'])

        # Apply categorisations 
        mdf = apply_outcome_categorisation(mdf)
        mdf = apply_crime_type_categorisation(mdf)

        # Renaming after categorisations
        mdf = rename_cols(mdf,{'Last outcome category':'Outcome','Final outcome':'Outcome category','postcode':'Postcode'})

        # Concat staged essex and kent property dataframes
        property_df = concat_property_csvs(essex_df, kent_df)

        # Save to CSV
        mdf.to_csv(PRIMARY_CRIME, index=False)
        property_df.to_csv(PRIMARY_PROPERTY, index=False)

        logging.info("Primary Layer completed successfully")
    except Exception as e:
        logging.error(f"Error during Primary Layer: {e}")

# Reporting layer
def reporting():
    """
    store the reported data to CSV files
    """
    logging.info("Starting Reporting layer")
    # ingest data
    crime_df = ingest_data(PRIMARY_CRIME)
    property_df = ingest_data(PRIMARY_PROPERTY)
    area_df = ingest_data(PRIMARY_COUNTY_AREA)
    try:
        # Preparing dataframe detailing burglaries + robberies per km^2 for each county each month 
        burg_rob_area_df = left_merge(crime_df,area_df,['County'])
        burg_rob_area_df = crime_category_filter(burg_rob_area_df,['Burglary + robbery'])
        burg_rob_area_df = group_by(burg_rob_area_df, ['County','Month'],{'Crime type category':'count','Area (km2)':'mean'})
        burg_rob_area_df = burg_rob_km2(burg_rob_area_df)
        burg_rob_area_df = del_cols(burg_rob_area_df,['Crime type category','Area (km2)'])
        burg_rob_area_df = df_reset_index(burg_rob_area_df)

        # Preparing dataframe detailing essex monthly data including property prices, sales and burglaries/ robberies per month
        essex_monthly_crime_data = county_filter(crime_df, ['Essex'])
        essex_monthly_burg_rob_data = crime_category_filter(essex_monthly_crime_data, ['Burglary + robbery'])
        essex_monthly_burg_rob_data = group_by(essex_monthly_burg_rob_data, ['Month'], {'Crime type category':'count'})
        essex_monthly_burg_rob_data = rename_cols(essex_monthly_burg_rob_data, {'Crime type category':'Count of burglaries/ robberies'})

        essex_monthly_property_data = county_filter(property_df, ['Essex'])

        essex_monthly_data = left_merge(essex_monthly_burg_rob_data, essex_monthly_property_data, ['Month'])

        # Preparing dataframe detailing kent monthly data including property prices, sales and burglaries/ robberies per month

        kent_monthly_crime_data = county_filter(crime_df, ['Kent'])
        kent_monthly_burg_rob_data = crime_category_filter(kent_monthly_crime_data, ['Burglary + robbery'])
        kent_monthly_burg_rob_data = group_by(kent_monthly_burg_rob_data, ['Month'], {'Crime type category':'count'})
        kent_monthly_burg_rob_data = rename_cols(kent_monthly_burg_rob_data, {'Crime type category':'Count of burglaries/ robberies'})

        kent_monthly_property_data = county_filter(property_df, ['Kent'])

        kent_monthly_data = left_merge(kent_monthly_burg_rob_data, kent_monthly_property_data, ['Month'])

        # Preparing dataframe detailing essex + kent top 5 postcodes with highest burglaries + robberies 
        postcode_crime = split_postcode(crime_df)
        postcode_crime = drop_null_rows(postcode_crime,['Postcode start'])
        postcode_crime = county_filter(postcode_crime, ['Essex','Kent'])
        postcode_crime_top = crime_category_filter(postcode_crime, ['Burglary + robbery'])
        postcode_crime_top = group_by(postcode_crime_top, ['Postcode start'], {'Crime ID': 'count'})
        postcode_crime_top = rename_cols(postcode_crime_top, {'Crime ID':'Count of burglaries/ robberies'})
        postcode_crime_top = sort_dataframe(postcode_crime_top, 'Count of burglaries/ robberies', asc= False)
        postcode_crime_top = postcode_crime_top.head()
        postcode_crime_top = df_reset_index(postcode_crime_top)
        
        postcode_crime_new = drop_dup_rows(postcode_crime,['County','Postcode start'])
        postcode_crime_new = del_cols(postcode_crime_new,['Crime ID','Month','Longitude','Latitude','Location','Outcome','Crime type','Outcome category','Crime type category','Postcode'])
        
        postcode_crime_top = left_merge(postcode_crime_top, postcode_crime_new, ['Postcode start'])
        postcode_crime_top = rename_cols(postcode_crime_top, {'Postcode start':'Postcode'})


        # Save to csv
        burg_rob_area_df.to_csv(REPORTING_BURGLARY_ROBBERY_DENSITY, index = False)
        essex_monthly_data.to_csv(REPORTING_ESSEX_MONTHLY_DATA, index= False)
        kent_monthly_data.to_csv(REPORTING_KENT_MONTHLY_DATA, index= False)
        postcode_crime_top.to_csv(REPORTING_POSTCODE_CRIME, index=False)

        logging.info("Reporting Layer completed successfully")
    except Exception as e:
        logging.error(f"Error during Reporting Layer: {e}")


def main(pipeline='all'):
    logging.info("Pipeline execution started")

    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                # If only staging is requested, print success and return
                logging.info("Pipeline run complete")
                return
            # Process the staged data
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                # If only primary is requested, print success and return 
                logging.info("Pipeline run complete")
                return
            # Generate reports based on processed data
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete")
                return
            logging.info("Full pipeline run complete")
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()








        

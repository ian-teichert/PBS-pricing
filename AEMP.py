# data management libraries
import pandas as pd                # database manipulation
import numpy as np                 # array manipulation / math functions
import json                        # read and write log files 
from zipfile import ZipFile        # manage and use zip files in Python

# data scraping libraries
import requests
from requests_html import HTML

# useful coding libraries
from pathlib import Path           # managing filepaths in readable fashion  
from urlpath import URL            # managing URLs in readable fashion
import string                      # contains some obscure Python string functions
from io import BytesIO             # used for reading/writing zip files
from sys import exit               # exit script execution prematurely

# progress bar
from tqdm.notebook import tqdm     # progress bar functionality 


class PricingDB:
    """
    PBS pricing database object. 
    Contains the methods for updating the pricing database and loading it for queries.

    BASE_DIR: Directory of the existing pricing database and logs. Will be created if this
    directory does not already exist.
    """

    def __init__(self, BASE_DIR):
        """
        Initiates PricingDB object.
        Contains two hidden variables (see below)

        _BASE_URL: Pharmaceutical Benefits Schedule (PBS) homepage URL('https://www.pbs.gov.au')
        __exman_prices_url: Stem URL which links to the ex-manufacturer price spreadsheets
        published by the PBS (https://www.pbs.gov.au/pbs/industry/pricing/ex-manufacturer-price)
        """

        self.BASE_DIR = Path(BASE_DIR)
        self._BASE_URL = URL('https://www.pbs.gov.au')
        self._exman_prices_url = self._BASE_URL / 'info/industry/pricing/ex-manufacturer-price'


    def run(self, debug_mode=False):
        """
        Checks for updates to the existing pricing database compared to the currently published
        PBS ex-manufacturer spreadsheets. If new data is found, merges this with the database
        and writes this to BASE_DIR.

        debug_mode: If set as true, the whole process will run but the resulting database is not
        written to BASE_DIR.
        """

        # Check for new data on the PBS that isn't already contained in the existing database
        # If any new data is found, this is returned as a Pandas dataframe (pd.DataFrame)
        new_data, log = self._download_updates()

        # Pythonic way to check if 'new_data' is a dataframe
        # If it is not, we have no new data, and we can exit the updating process
        if not isinstance(new_data, pd.DataFrame):
            print('No new data found.')
            return

        # Load the latest data from the existing database. Relationships between SKUs will be
        # based on this latest data alone.
        old_data = self.load_db(latest_month_only=True)

        # Find earliest date in new data and oldest date in existing data.  
        earliest_date = new_data.Date.min()
        oldest_date = old_data.Date.max()

        # If 'load_db' does not return a dataframe, there is no existing database.
        # Alternatively, if oldest_date >= earliest_date, some error has occurred,
        # and we do not want to append in this instance. We can delete old_data.
        if not isinstance(old_data, pd.DataFrame) or not oldest_date < earliest_date:
            APPEND = False
            df = new_data
            del old_data
        else:
            APPEND = True
            df = pd.concat([old_data, new_data], join='inner', sort=False)


        # We now have our working dataframe 'df'
        # Run the lookup which connects the datasets together.
        df = self._perform_lookup(df)

        # If appending to the existing database, remove the earliest date in 'df'
        # (as this is already in the existing database)
        if APPEND:
            df = df.loc[df.Date != df.Date.min()]
            
        # If in debug_mode, just return the dataframe (df) 
        if debug_mode:
            return df

        # Otherwise, write the data to the pricing database in BASE_DIR and log all
        # downloaded data        
        else:
            self.write_db(df, append=APPEND)
            self._write_download_log(log)

        print('Complete.')


    def _read_download_log(self):
        """
        Returns the logged URL stems that have already been imported into the
        pricing database, which are stored as a list of strings in a json file.
        """
        
        # Create path to logfile and make sure the directory exists.
        path = Path(self.BASE_DIR) / 'logs' / 'download_log.json'
        path.parent.mkdir(parents=True, exist_ok=True)

        # If the logfile exists, read it, otherwise return an empty list.
        if path.exists():
            with open(path, 'r') as log:
                return json.load(log)
        else:
            return []

        
    def _write_download_log(self, log_data):
        """
        Write the URL stems that have been imported into the pricing database
        to the download logfile, stored as a list of strings in a json file.
        """
        
        # Create path to logfile and make sure the directory exists.
        path = Path(self.BASE_DIR) / 'logs' / 'download_log.json'
        path.parent.mkdir(parents=True, exist_ok=True)

        # Open logfile and dump the list into it.
        with open(path, 'w') as log:
            json.dump(log_data, log)
        
        
    def _check_for_updates(self):
        """
        Checks the current exman-prices download page for new data that isn't stored in the
        download log. Returns a list of URL stems that need to be imported.
        """
        
        # Read logfile to find previously downloaded/imported data
        download_log = self._read_download_log()

        # Scrape the ex-man prices PBS page for new content
        r = requests.get(self._exman_prices_url)
        html = HTML(html=r.content)

        # The below is a 'list comprehension' which filters the URLs on the ex-man prices webpage
        return [
        ele.attrs['href'] for ele in html.find('a')
        if (ele.attrs['href'].lower().__contains__('xls')) and
        (ele.attrs['href'] not in download_log)
        ]
        
        # In expanded form, it looks like this
        # res = []
        # for ele in html.find('a'):
        #    if ele.attrs['href'].lower().__contains__('xls') and ele.attrs['href'] not in download_log:
        #         res.append(ele.attrs['href'])
        # return res

        # On a webpage, each URL has an 'a' tag <a href='...'>
        # The above looks for all the URLs on the ex-man prices webpage and checks if they have 'xls' in their path
        # If 'xls' is in the path AND it is not already in the download_log, then we append it to the 'res' list
        # Once we have found all the URLs stems, we return the 'res' list


    def _read_exman_source(self, url):
        """
        Reads source exman .xls file, including date and schedule from url,
        and returns a cleaned df.
        Includes two self-defined functions specific to reading data from the URL.
        """

        def read_exman_date(url):
            """
            Returns the date of the dataset passed in the url in datetime format
            """

            url = URL(url)
            return pd.to_datetime('-'.join(url.stem.split('-')[-3:]))

            # Converts '/industry/pricing/ex-manufacturer-price/2021/ex-manufacturer-prices-efc-2021-03-01.XLSX'
            # To '2021-03-01' in datetime format.


        def read_exman_schedule(url):
            """
            Returns the schedule of the dataset passed in the url (efc or non-efc)
            """

            url = URL(url)

            if url.stem.partition('prices-')[2].partition('-')[0].lower() == 'efc':
                return 'efc'
            else:
                return 'non-efc'

            # Converts '/industry/pricing/ex-manufacturer-price/2021/ex-manufacturer-prices-efc-2021-03-01.XLSX'
            # To 'efc' (or 'non-efc' if relevant)
            # Return efc/non-efc depending on result


        # Read date and schedule from URL
        date = read_exman_date(url)
        schedule = read_exman_schedule(url)

        # Load df from URL using Pandas read_excel
        df = pd.read_excel(url)

        # Store date and schedule in df
        df['Date'] = date
        df['Schedule'] = schedule

        # Return the cleaned dataframe
        return self._clean_df(df, schedule)


    class AMTNameError(Exception):
        pass

        # Specific exception that is raised if there is an error in AMT names
        # during the clean_df method.


    def _clean_df(self, df, schedule):
        """
        Cleans the dataframe passed through it, with different actions based on schedule  
        """
        
        def catch_amt_name(col):
            """
            Used to catch the AMT trade product pack (TPP) name for standardisation,
            as this changes over time.
            If any names in 'catch' are found, return True, otherwise returns False
            """
            
            catch = ('amt', 'tpp', 'product pack')
            return any(x in col.lower() for x in catch)
        
        
        # Clean trailing or leading whitespace in field names
        df.columns = [col.strip() for col in df.columns]
        
        # As the name for the 'AMT name' field changes over time, we need to identify the
        # appropriate column
        AMT_name = list(filter(catch_amt_name, df.columns))
        
        # Raise an error if the AMT name field is not found OR if multiple fields are found
        # (there should only be one)
        if not AMT_name:
            raise AMTNameError(f'AMT name not found')
        if len(AMT_name) > 1:
            raise AMTNameError('len(AMT_name) > 1')
        
        # AMT_name is currently a list. Take the first and only element to get the string
        # Rename the AMT name field as 'AMT TPP' - standardises the column name across time
        AMT_name = AMT_name[0]
        df.rename(columns={AMT_name: 'AMT TPP'}, inplace=True)
            
        # Specific edge-case catching
        if 'C\'wlth Pays Premium' in df.columns:
            df.rename(columns={'C\'wlth Pays Premium': 'Commonwealth Pays Premium'}, inplace=True)

        # Rename columns in EFC/Non-EFC schedule data to allow for their merging
        # e.g. EFC data has 'DPMA' (dispensed price for maximum amount) rather than 'DPMQ' field
        # (dispensed price for maximum quantity)
        rename_cols = {
            'Maximum Amount': 'Maximum Quantity/Amount',
            'Maximum Quantity': 'Maximum Quantity/Amount',
            'Number Repeats': 'Maximum/Number Repeats',
            'Maximum Repeats': 'Maximum/Number Repeats',
            'DPMA': 'DPMQ/DPMA',
            'DPMQ':'DPMQ/DPMA',
            'Claimed DPMA': 'Claimed DPMQ/DPMA',
            'Claimed DPMQ':'Claimed DPMQ/DPMA'
        }
        
        # Filter the rename_cols dictionary to only columns which are in the dataframe
        # Perform the renaming
        rename_cols = {k: v for k, v in rename_cols.items() if k in df.columns}
        df.rename(columns=rename_cols, inplace=True)

        # Depreceated/Misnamed columns to drop from final dataframe
        columns_to_drop = [
            'index', 'AMT Trade Product Pack Pack', 'Exempt', 'Therapeutic Group',
            'New PI or Brand', 'Previous Pricing Quantity', 'Previous AEMP', 'Price Change Event',
            'Previous Premium', 'ATC', 'DD', 'MRVSN', 'Substitutable', ' Item Code',
            'Authorised Rep', 'Email', 'AMT Trade Product pack', 'AMT Trade product Pack',
            'ANT Trade Product Pack', 'TPP', 'AMT Trade Product Pack ', 'Amt Trade Product Pack'
        ]
            
        # Filter to only the columns that are present in the current dataframe
        # __contains__ is a handy dunder/magic that performs the opposite of 'in'; e.g.,
        # 'o' in 'hello' == True
        # 'hello'.__contains__('o') == True
        # Python string objects have the __contain__ method, as do many other types
        # filter(function, iterable) - creates a generator which returns True/False for each element
        # in iterable after passing it through function

        # This may be able to be simplified to a simple list comprehension (below) however I have not tested it
        # columns_to_drop = [field for field in df.columns if field not in columns_to_drop]
        columns_to_drop = [field for field in filter(columns_to_drop.__contains__, df.columns)]

        # Standard dataframes have two axes - the rows/index (axis=0) and the columns (axis=1)
        # Field names are stored in columns, so we specify this axis
        # Setting inplace=True means we don't need to specify df = df.drop(...)
        df.drop(columns_to_drop, axis=1, inplace=True)
            
        # Set commonwealth pays premium field to a boolean
        # Replace all 'Yes' with True and 'No'/NaN with False
        if 'Commonwealth Pays Premium' in df.columns:
            df['Commonwealth Pays Premium'] = df['Commonwealth Pays Premium'].apply(
                lambda x: True if x == 'Yes' else False
            )
        
        # Clean up premium field
        # Creates a premium type column (any letters which aren't associated with the premium amount)
        # And a premium column (the actual premium ammount)
        float_chars = '1234567890.'
        if 'Premium' in df.columns:
            df['Premium Type'] = df['Premium'].apply(lambda x: ''.join(filter(string.ascii_letters.__contains__, str(x))))
            df['Premium'] = df['Premium'].apply(lambda x: ''.join(filter(float_chars.__contains__,str(x))))

        return df


    def _download_updates(self, debug_mode=False):
        """
        Downloads any missing datasets from the PBS and returns these as a df
        """

        # Check for any new datasets needed. If none found, returns (None, None)
        # Returns a tuple as the method's which call _download_updates check for
        # a) a dataframe and b) a log of downloaded URLs
        dataset_urls = self._check_for_updates()
        if not dataset_urls:
            return None, None

        # Stores new datasets as dataframes in a list 'new_data'
        # Later we can concatenate all of these datasets in one go using pd.concat()
        # tqdm allows for visualisation of the progress when used in Jupyter notebooks
        new_data = []
        for url in tqdm(dataset_urls, desc='Importing new data'):
            new_data.append(self._read_exman_source(self._BASE_URL / url))

        # Set sort to false, otherwise it performs computationally expensive sort function
        # which is not useful
        df = pd.concat(new_data, sort=False)
        
        # Create a new log of updates - extends the list of existing logged URLs with the new URLs 
        # This is not yet written to the logfile - this is only performed once the database is updated.
        log = self._read_download_log()
        log.extend(dataset_urls)

        # Each new dataset will be indexed from 0 -> n, where n is number of rows in the dataset
        # These are maintained in the concatenated df, so we reset_index to restore the natural 0 -> n order
        # By default, reset_index retains the original index as a new column. Setting drop=True means
        # we can drop this from our df.
        return df.reset_index(drop=True), log
        

    def load_db(self, latest_month_only=False, name='db'):
        """
        Loads pricing database for running queries or updating the database.
        By default the name of the database is 'db'. This can be changed but it is not recommended. 
        """
        
        # Check if database exists.
        path = self.BASE_DIR / name
        if not path.exists():
            return None
        
        # Read the database and convert the Date field to datetime
        df = pd.read_feather(path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # If only the latest month is required, we can filter the df to just the latest month
        if latest_month_only:
            df = df.loc[df.Date == df.Date.max()]
        
        return df


    def write_db(self, df, append=False, name='db'):
        """
        Writes dataframe to local pricing database, or appends to existing
        """
        
        # Ensures index is in natural 0 -> n format, where n = number of rows
        df.reset_index(drop=True, inplace=True)
        
        # If writing new database, write straight to BASE_DIR
        if not append:
            df.to_feather(self.BASE_DIR / name)

        # Otherwise, load the existing df as df_base and append the new data together
        # Reset index and write to BASE_DIR
        else:
            df_base = self.load_db()
            df = pd.concat([df_base, df], sort=False)
            df.reset_index(drop=True).to_feather(self.BASE_DIR / name)

        print(f'Database has been written to {self.BASE_DIR / name}')

        # The database is stored in feather format, which is much faster for reading/writing data
        # than traditional .csv formats and performs well against other feasible types.
        # See https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d

    def _perform_lookup(self, df):
        """
        Generate Unique SKU IDs for generating previous AEMPs and creating longitudinal relationships
        """

        # Instantiate PBSData object, which can read useful PBS data including ATC maps and PBS item drug maps 
        pbs_data = PBSData()

        # These columns are sufficient to uniquely identify a SKU
        # AMT TPP contains Brand, formulation, and quantity
        lookup_cols = [
            'Item Code', 'AMT TPP', 'Pack Quantity', 'Pricing Quantity', 
            'Vial Content', 'Maximum Quantity/Amount', 'Number/Maximum Repeats'
        ]

        # Filter to cols that are present in df
        # Aggregates each of the relevant columns using '-' as a separator
        # This gives a SKU ID which persists over time
        for i in tqdm(range(1), desc='Generating SKU ID'):
            lookup_cols = [col for col in lookup_cols if col in df.columns]
            df['SKU ID'] = df[lookup_cols].applymap(str).agg('-'.join,axis=1)

        # Lookup previous AEMP. Fill value is NaN - if none is found, then there was no previous AEMP
        for i in tqdm(range(1), desc='Calculating previous prices'):

            # Group the df by SKU ID and return its AEMP. Shift AEMP values forward one month and store this
            # in the newly created 'Previous AEMP' field. This has the effect of storing the previous month's
            # AEMP in the current month's row by SKU ID
            df['Previous AEMP'] = df.groupby('SKU ID')['AEMP'].shift(fill_value=np.nan)

            # Create fields for AEMP increases and AEMP decreases (True or False)
            conditions_increase = [
              (df['AEMP'] > df['Previous AEMP']) & (df['Previous AEMP'] != np.nan),
              (df['AEMP'] <= df['Previous AEMP']) & (df['Previous AEMP'] != np.nan)
            ]
            conditions_decrease = [
              (df['AEMP'] < df['Previous AEMP']) & (df['Previous AEMP'] != np.nan),
              (df['AEMP'] >= df['Previous AEMP']) & (df['Previous AEMP'] != np.nan)
            ]

            outcomes = [True, False]

            # Numpy - Across the df, select from the cases in outcomes (True or False) based
            # on whether the conditions for an AEMP increase or decrease are met/not met
            df['AEMP_Increase'] = np.select(conditions_increase, outcomes, default=False)
            df['AEMP_Decrease'] = np.select(conditions_decrease, outcomes, default=False)

            # Absolute and relative change calculations
            df['AEMP_Abs_Change'] = df['AEMP'] - df['Previous AEMP']
            df['AEMP_Rel_Change'] = (df['AEMP'] - df['Previous AEMP']) / df['Previous AEMP']

        for i in tqdm(range(1), desc='Merging with PBS Item Map'):
            # The item drug map contains the associated ATC code for each PBS item - we want to store this in the df
            # However the PBS item drug map stores PBS item codes as 6 digit item codes, rather than variable lengths
            # on the ex-man prices spreadsheets, padding from the left using '0'.
            # Here, we create an item code which is of length 6 for merging with the PBS item drug map
            df['ItemCodeLookup'] = df['Item Code'].str.rjust(6, '0')

            # Merge the datasets. Keep all rows in the left df and match on the 6-char item codes
            df = df.merge(pbs_data.item_map[['ITEM_CODE','ATC_Code']], 
                how='left', 
                left_on='ItemCodeLookup', 
                right_on='ITEM_CODE')

            # Drop the 6-char item code fields and rename ATC code field name
            df = df.drop(['ITEM_CODE', 'ItemCodeLookup'],axis=1)
            df.rename(columns={'ATC5_Code':'ATC_Code'},inplace=True)

        for i in tqdm(range(1), desc='Generating ATC labels'):
            # Generate leveled ATC code labels by matching with the ATC map on
            # 1st character
            # 1st-3rd characters
            # 1st-4th characters
            # 1st-5th characters
            # For each of these levels, match the existing dataframe with the ATC label from the ATC map
            
            # Create levelled ATC codes
            df['ATC1'] = df['ATC_Code'].str[0]
            df['ATC3'] = df['ATC_Code'].str[:3]
            df['ATC4'] = df['ATC_Code'].str[:4]
            df['ATC5'] = df['ATC_Code'].str[:5]

            for ATC_level in ['ATC1','ATC3','ATC4','ATC5','ATC_Code']:
                
                # Merge on each ATC level
                df = df.merge(pbs_data.atc_map[['ATC Code', 'Label']],
                              how='left',
                              left_on=ATC_level,
                              right_on='ATC Code').drop('ATC Code', axis=1)

                # Create label name in df
                df.rename(columns={'Label': ATC_level + '_label'}, inplace=True)

        return df


    def save_latest_month(self):
        """
        Saves the latest month of data to the BASE_DIR / Monthly Data directory for uploading to Salesforce.
        """
        
        # Load latest month of data
        df = self.load_db(latest_month_only=True)

        # Ensure the directory for saving monthly data exists
        SAVE_DIR = self.BASE_DIR / 'Monthly Data'
        SAVE_DIR.mkdir(exist_ok=True)
        
        # Convert datetime date from dataframe into string format for writing to SAVE_DIR
        # Create save_path
        date = df.Date.max().strftime('%Y-%m-%d')
        save_path = SAVE_DIR / (date + '_AEMP data.csv')

        # Double-check that only the latest month of data is saved. Write as .csv (for uploading to Salesforce)
        # No index is required in .csv format. This must be kept as false or this will cause an error in Salesforce
        df.loc[df.Date == df.Date.max()].to_csv(save_path, index=False)

        print(f'Saved to Google Drive: {save_path}')


    def save_specific_months(self, start_date, end_date):
        """
        Saves user-defined months of data to the BASE_DIR / Monthly Data directory as a .csv for uploading to Salesforce.
        start_date and end_date: Enter in the format 'yyyy-mm' (including quotes). Do not enter a day.
        """

        # Convert state_date/end_date to datetime format
        timestamp_start = pd.to_datetime(start_date + '-01')
        timestamp_end = pd.to_datetime(end_date + '-01')

        # Ensure save_path directory exists and create path to saved data
        SAVE_DIR = self.BASE_DIR / 'Monthly Data'
        SAVE_DIR.mkdir(exist_ok=True)
        save_path = SAVE_DIR / (start_date + '_to_' + end_date + '_AEMP data.csv')

        # Load database and filter to required dates. Save resulting df as a .csv without an index.
        # No index is required in .csv format. This must be kept as false or this will cause an error in Salesforce
        df = self.load_db()
        df.loc[(df.Date >= timestamp_start) & (df.Date <= timestamp_end)].to_csv(save_path, index=False)

        print(f'Saved to Google Drive: {save_path}')


class PBSData:
    """
    Wrapper for methods to grab various data from the PBS website
    """
    
    def __init__(self):
        """
        _BASE_URL: Pharmaceutical Benefits Schedule (PBS) homepage URL('https://www.pbs.gov.au')
        source_url: PBS download source page which contains PBS text files, including most recent ATC maps
        item_drug_map_url: Link to PBS item drug map csv file
        _exman_prices_url: Stem URL which links to the ex-manufacturer price spreadsheets
        published by the PBS (https://www.pbs.gov.au/pbs/industry/pricing/ex-manufacturer-price)
        """

        self._BASE_URL = URL('https://www.pbs.gov.au')
        self.source_url = self._BASE_URL / 'info/browse/download'
        self.item_drug_map_url = self._BASE_URL / '/statistics/dos-and-dop/files/pbs-item-drug-map.csv'
        self._exman_prices_url = self._BASE_URL / 'info/industry/pricing/ex-manufacturer-price'

        for i in tqdm(range(1), desc='Loading PBS data'):
            self.text_files_zip = self.get_latest_PBS_text_files()
            self.atc_map = self.get_atc_from_text_files(self.text_files_zip)
            self.item_map = self.get_item_drug_map()

        
    def get_latest_PBS_text_files(self):
        """
        Returns a ZipFile of the most recent PBS text files
        """

        # Grab HTML data from source_url
        r = requests.get(self.source_url)
        html = HTML(html=r.content)

        # Filter to current PBS text files .zip
        # Checks if 'PBS Text Files' is in the URL title and if .zip is in the file
        # We know the first element found is the most recent, so we grab the first element with [0]
        # Finally we want the 'href' attribute of the element - this is the URL stem we can download
        href = [ele for ele in html.find('a.xref') 
                         if ('PBS Text files' in ele.attrs['title'])
                         and ('.zip' in ele.attrs['href'].lower())][0].attrs['href']
    

        # String manipulation of the URL to conform with the PBS BASE_URL format
        discard, sep, url = href.partition('downloads')
        zip_url = sep + url

        # Load the zipfile as bytes input/output format. Allows us to read the data without saving to file.
        r = requests.get(self._BASE_URL / zip_url, stream=True)
        return ZipFile(BytesIO(r.content))


    def get_atc_from_text_files(self, zipfile):
        """
        Gets ATC code map from PBS text files and returns as df with cols ('ATC', 'Description')
        """

        # Look through the zipfile directory and find the file starting with 'atc_'
        zipfile_dir = zipfile.namelist()
        for file in zipfile_dir:
            if 'atc_' in file:
                
                # We have found the relevant file - we can break out of the for loop
                # The filename is currently stored in the 'file' variable

                break

        # Open the file and read it. Must parse utf-8 string format and separate columns using the '!' separator
        # We can ignore the first line (using [1:]) as we will make our own column names (ATC Code and Label)
        with zipfile.open(file) as f:
            data = [line.decode('utf-8').strip().split('!') for line in f.readlines()][1:]

        # Return the dataset as a dataframe
        return pd.DataFrame(data, columns=['ATC Code', 'Label'])


    def get_item_drug_map(self):
        """
        Returns a df of the PBS item drug map
        """

        # Read the csv with 'latin-1' encoding. Rename the columns and return the dataframe
        df = pd.read_csv(self.item_drug_map_url, encoding='latin-1')
        df.columns = ['ITEM_CODE', 'DRUG_NAME', 'PRESENTATION', 'ATC_Code']

        return df
    

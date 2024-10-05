import sys
import requests
import os
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

base_url = "https://api.census.gov/data/"
base_query = "get=ESTAB&for=zipcode:*&"

# naics_codes = ['11', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81'#, '92'
#                ]

naics_codes = ['52', '54', '55', '62', '71', '72'
               ]

naics_code_mapper = codes = {'31-33': '31', '44-45': '44', '48-49': '48'}

def requests_retry_session(retries=5, backoff_factor=1, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_zip_code_business_pattern_data(year, data_output_path):
    # for years between 2017 and 2022, use NAICS2017
    # for years between 2012 and 2016, use NAICS2012
    # for years between 2008 and 2011, use NAICS2007
    # for years between 2003 and 2007, use NAICS2002
    # for years between 1998 and 2002, use NAICS1997
    
    naics_dfs = []

    if year >= 1998 and year <=2018:
        trailing_url = "/zbp?"

    elif year >= 2019 and year <=2022:
        trailing_url = "/cbp?"
        
    if year >= 1998 and year <= 2002:
        naics = "NAICS1997="
        
    elif year >= 2003 and year <= 2007:
        naics = "NAICS2002="
        
    elif year >= 2008 and year <= 2011:
        naics = "NAICS2007="
        
    elif year >= 2012 and year <= 2016:
        naics = "NAICS2012=" 
        
    elif year >= 2017 and year <= 2022:
        naics = "NAICS2017="
        

    for naics_code in naics_codes:
        code = naics_code

        if year <= 2011:
            code = naics_code_mapper.get(naics_code, naics_code)
    
        final_url = base_url+str(year)+trailing_url+base_query+naics+code
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
        
        response = requests_retry_session().get(final_url, headers=headers)

        data = response.json()
        data_header = data.pop(0)
        
        df = pd.DataFrame(data, columns=data_header)
        naics_dfs.append(df)

    pd.concat(naics_dfs).to_csv(data_output_path, index=False, compression="gzip")

def is_exists(output_path):
    return os.path.exists(output_path)

def is_valid_args(args):
    if len(args) != 3:
        raise ValueError(f"Invalid number of arguments. Expected 3 but got {len(args)}")
    
    for i in range(2):
        if i == 0:
            year_type = 'start'
        else:
            year_type = 'end'
            
        try:
            year = int(args[i])
        except ValueError:
            raise TypeError(f"Invalid input type for year {year_type}. Expected input of type 'int'")
        
        if int(args[i]) != float(args[i]):
            raise ValueError(f"Invalid input value for year {year_type}. Expected year {year_type} to be a natural number")
    
        elif int(args[i]) < 1998 or int(args[i]) > 2022:
            raise ValueError("Invalid year range! Please, valid range is between 1998 and 2022. Those are the years I can do for now.")

    if int(args[0]) > int(args[1]):
        raise ValueError("Invalid year range! Expected Year Start < Year End, but got Year Start > Year End.")

    return True

def main(args):
    
    if is_valid_args(args):
    
        output_dir = args[2]
        if not is_exists(output_dir):
            os.makedirs(output_dir)

        for year in range(int(args[0]), int(args[1])+1):
            data_output_path = os.path.join(output_dir, str(year)+".csv")

            if not is_exists(data_output_path):
                print(f"getting data for year {year}")
                get_zip_code_business_pattern_data(year, data_output_path)
                print(f"Done getting data for year {year} with success!")

            else:
                print(f"Data for {year} already exists, skipping it!")
                continue

        print("All tasks completed sucessfully!")
        
if __name__ == "__main__":
    main(sys.argv[1:])
    
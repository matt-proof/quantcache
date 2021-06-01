###########################
"""Data Storage and Retrieval"""
###########################

import warnings
import pandas as pd
import numpy as np
from collections.abc import Iterable
import inspect
import os
import shutil
import datetime as dt
import re
import functools

"""
TO DO:

Necessary:

- Write testing script with pytest
- Write documentation
- Write examples
- Fix Problems where argument is list of lists
- Function for setting file location

Nice-to-Have

- Support flexible database writing
- Remember index names
- Other data types?
- use iPython widgets for choosing data
- Mutliple dataframe outputs
- query nans one-by-one, raise appropriate errors
- delete nan rows at the end
- Add comments to dataframe writing, print out
"""

###########################
"""Exceptions"""
###########################

class CacheException(Exception):
    pass

###########################
"""Registration"""
###########################

def is_data(val):
    """Determines if @val is a data or float object"""

    return isinstance(val,(np.ndarray, pd.core.base.PandasObject,float))

def is_registrable(val):
    """Determines if @val is well-suited for data registration"""

    # print(val)
    # print(isinstance(val,Iterable))
    if (not isinstance(val,Iterable)) and pd.isna(val):
        return True

    if is_data(val):
        return False
    if isinstance(val,Iterable):
        for elt in val:
            if is_data(elt):
                return False
    return True

def registry_prep(val):
    """Prepares registrable @val for registration

    Sets are stored as lists and dicts as lists of tuples due to ordering issues
    """

    if not is_registrable(val):
        raise CacheException(f'Object {val} is not registrable')
    if isinstance(val,set):
        val = [registry_prep(elt) for elt in sorted(val)]
    if isinstance(val,dict):
        val = [(key, registry_prep(elt)) for key, elt in sorted(val.items())]
    if callable(val) and hasattr(val,'__name__'):
        val = val.__name__
    if isinstance(val,list):
        val = str([registry_prep(elt) for elt in val])

    return val

def prep_dict_for_registry(info_dict):
    """Prepares a dictionary for queries in the registry"""

    dict_for_registry = {key: registry_prep(val) for key, val in info_dict.items() if is_registrable(val)}

    for key, val in dict_for_registry.items():
        if not isinstance(val,int):
            dict_for_registry[key] = str(val)

    return dict_for_registry

###########################
"""Registry Displays"""
###########################

def display_registry_uniques_old(registry):
    """Displays the columns of the registry that have multiple unique values"""

    has_uniques = False
    for col in registry.columns:
        uniques = registry[col].unique()
        if col not in {'Date', 'Data Number'} and uniques.shape[0] > 1:
            print(f'\n{col} options:')
            print(tuple(uniques))
            has_uniques = True

    return has_uniques

def display_registry_uniques(registry):
    """Displays the columns of the registry that have multiple unique values"""

    unique_cols = []
    for col in registry.columns:
        uniques = registry[col].unique()
        if col not in {'Date', 'Data Number', 'Runtime'} and uniques.shape[0] > 1:
            unique_cols.append(col)

    registry = registry.set_index('Data Number')

    reg_options = registry[unique_cols].drop_duplicates()

    has_uniques = bool(unique_cols)

    if has_uniques:
        print('Registry Options')
        display(reg_options)

    return has_uniques

def get_registry_uniques(parent_dir):
    '''Displays the columns in the registry with multiple unique values'''

    registry = get_registry(parent_dir)

    has_uniques = display_registry_uniques(registry)

    if not has_uniques:
        print('Registry has no fields with multiple unique values')

def show_data_specs(parent_dir,data_number):
    """Gets the data specifications for a specific data number and parent_dir"""

    registry = get_registry(parent_dir)
    registry.set_index('Data Number',inplace = True)
    specs = registry.loc[data_number]
    for index, item in specs.iteritems():
        print(f'{index} = {item}')

###########################
"""Querying"""
###########################

def retrieve_info_row(info_dict, registry, assume_nones = False, raise_errors = False, suppress = False,
                      allow_duplicates = False, parent_dir = None):
    """Gets row from @registry specified by @info_dict

    @assume_nones assumes that every value not specified by @info_dict
    is None. If set False, will display helpful information if duplicate
    queries are found, and raise an error.
    """

    # show_all(locals())


    if parent_dir is None and not suppress:
        raise CacheException('Specify parent_dir if not suppressing messages')

    if not set(info_dict.keys()).issubset(set(registry.columns)):
        parent_dir_str = parent_dir + ' ' if parent_dir is not None else ''
        msg = f'Fields {set(info_dict.keys()).difference(set(registry.columns))} not found in {parent_dir_str}registry'
        if raise_errors:
            raise CacheException(msg)
        if not suppress:
            print(msg,'\n')

        return pd.DataFrame({})

    if assume_nones:
        nones_list = [col for col in registry.columns if col not in list(info_dict.keys()) + ['Date','Data Number','Runtime']]
    else:
        nones_list = []
    query_dict = {}
    for key, val in info_dict.items():
        if is_registrable(val):
            if (not isinstance(val,Iterable)) and pd.isna(val) and val is not None:
                nones_list.append(key)
            else:
                query_dict[key] = registry_prep(val)

    if nones_list:
        registry_query = registry.copy()[registry[nones_list].isna().all(axis = 1)]
    else:
        registry_query = registry.copy()

    # show_all(locals())

    for key, val in query_dict.items():
        registry_query = registry_query[(registry_query[key] == str(val)) | (registry_query[key] == val)]
        if registry_query.empty:
            msg = f'Data not found in {parent_dir}. Registry query closed on {key} = {val}'
            if raise_errors:
                raise CacheException(msg)
            if not suppress:
                print(msg, '\n')
            break

    if registry_query.shape[0] >= 2 and not allow_duplicates:
        msg = 'Duplicate datasets found'
        if parent_dir is not None:
            msg += f' in {parent_dir}'
        msg += f'\nData Numbers: {set(registry_query["Data Number"])}'
        print(msg, '\n')
        has_uniques = display_registry_uniques(registry_query)
        if not has_uniques:
            print('\nDuplicates are exact')
        raise CacheException(msg)
    return registry_query

def check_parent_dir(parent_dir,stack):
    """Checks to see if parent_dir is properly specified"""

    if parent_dir is None:
        ex = CacheException('Outer function name not found. Please specify parent_dir parameter')
        try:
            parent_dir = f'{stack[1].function}_data'
        except IndexError:
            raise ex
        if parent_dir in ['execfile','run_code']:
            raise ex

    return parent_dir

def get_registry_path(parent_dir):
    """Gets file path for registry"""

    return os.path.join(parent_dir,'registry.csv')

def get_registry(parent_dir):
    '''Gets the registry as a pandas DataFrame'''

    return pd.read_csv(get_registry_path(parent_dir))

def retrieve_data_filename(info_dict, stack = None, parent_dir = None, assume_nones = False,
                           raise_errors = True, suppress = False):
    """Gets filename for data in @parent_dir with specs given by @info_dict

    Returns filename if it exists, and None if not.
    """

    if stack is None:
        stack = inspect.stack()

    parent_dir = check_parent_dir(parent_dir,stack)

    # show_all(locals())

    if os.path.exists(parent_dir):
        registry = get_registry(parent_dir)
        info_row = retrieve_info_row(info_dict, registry, assume_nones = assume_nones, raise_errors = raise_errors,
                                     suppress = suppress, allow_duplicates = False, parent_dir = parent_dir)
        row_size = info_row.shape[0]
        if row_size == 0:
            return None, None, None
        else:
            data_number = info_row['Data Number'].iloc[0]
            date = info_row['Date'].iloc[0]
            if 'Runtime' in info_row.columns:
                runtime = info_row['Runtime'].iloc[0]
                if pd.isna(runtime):
                    runtime = None
            else:
                runtime = None
            filename = os.path.join(parent_dir,f'{data_number}.csv')
            return filename, date, runtime
    else:
        msg = f'Parent directory {parent_dir} not found'
        if raise_errors:
            raise CacheException(msg)
        if not suppress:
            print(msg, '\n')
        return None, None, None

def date_for_readout(date_str):
    """Prepares date string for readout"""

    stripped = re.match(r"(.*?)\.",date_str)
    if stripped is None:
        return date_str
    else:
        return stripped.group(1)

def default_index(df):
    """Checks if index of dataframe is default"""

    return df.index.name is None and not isinstance(df.index, pd.MultiIndex) or isinstance(df.index,pd.RangeIndex)

###########################
"""Main Write/Retrieve Functions"""
###########################

def retrieve_data(info_dict, parent_dir = None, assume_nones = False, suppress = False,
                  overwrite = False, raise_errors = True):
    """Gets data stored in in @parent_dir with specs given by @info_dict

    Returns data if it exists, and None if not.

    @suppress suppress printout

    If @overwrite, not retrieving data, so returns None
    """

    filename, date, runtime = retrieve_data_filename(info_dict, parent_dir = parent_dir, assume_nones = assume_nones,
                                      stack = inspect.stack(), raise_errors = raise_errors, suppress = suppress)

    if filename is None:
        return None
    if overwrite:
        if not suppress:
            print(f'Overwriting {filename}\n')
        return None
    else:
        if not suppress:
            print(f'Loading data from {filename}')
            print(f'Original write time: {date_for_readout(date)}')
            if runtime is not None:
                print(f'Original runtime: {date_for_readout(str(runtime))}')
            print()
        return pd.read_csv(filename)

def write_data(data,info_dict,overwrite = False, parent_dir = None, suppress = False, runtime = None):
    """Writes data to csv in @parent_dir, with filename corresponding to Data Number in registry

    @info_dict determines the specifics of the data for interpretation and identification

    If @overwrite, the data is overwritten if it exists

    Adds new column to registry if @data_dict key is not present as a column in the registry.
    In this case all other values for this field in the registry will be set to None.

    If it does not exist, creates @parent_dir and a registry.csv file that
    contains info about what files correspond to what dictionaries.

    Be sure to add @parent_dir to .gitignore if being created.
    """

    if not isinstance(data, pd.core.base.PandasObject):
        raise CacheException(f'Data must be a pandas object')

    dict_for_registry = prep_dict_for_registry(info_dict)

    parent_dir = check_parent_dir(parent_dir,inspect.stack())

    if os.path.exists(parent_dir):
        registry = get_registry(parent_dir)
        info_row = retrieve_info_row(dict_for_registry, registry, assume_nones = True,
                                     raise_errors = False, suppress = True, allow_duplicates = False,
                                     parent_dir = parent_dir)
        row_size = info_row.shape[0]

        if row_size == 1:
            if overwrite:
                data_number = info_row['Data Number'].iloc[0]
                registry = registry[registry['Data Number'] != data_number]
            else:
                return
        else:
            data_number = registry['Data Number'].max() + 1
        dict_for_registry['Data Number'] = data_number
        dict_for_registry['Date'] = dt.datetime.today()
        if runtime is not None:
            dict_for_registry['Runtime'] = str(runtime)
        new_row = pd.Series(dict_for_registry)
        for key in dict_for_registry:
            if key not in registry.columns:
                registry[key] = None
        registry = registry.append(new_row, ignore_index=True)
        registry = registry.sort_values(by = 'Data Number').set_index(['Data Number','Date'])

    else:
        if not suppress:
            print(f'Creating directory {parent_dir}')
        os.mkdir(parent_dir)
        data_number = 0
        dict_for_registry['Data Number'] = data_number
        dict_for_registry['Date'] = dt.datetime.today()
        registry = pd.DataFrame(pd.Series(dict_for_registry)).T.set_index(['Data Number','Date'])

    registry.to_csv(get_registry_path(parent_dir))

    index_is_default = default_index(data)
    filename = os.path.join(parent_dir,f'{data_number}.csv')

    if not suppress:
        print(f'Saving data to {filename}')
        print(f'Write time: {date_for_readout(str(dict_for_registry["Date"]))}')

        if runtime is not None:
            print(f'Runtime: {date_for_readout(str(runtime))}')
        print()
    data.to_csv(filename,index = not index_is_default)

###########################
"""Main Decorator Function Helpers"""
###########################

def runtime_decor(func):
    """Decorator for passing back runtime of function as timedelta object"""

    def func_with_runtime(*args,**kwargs):

        start_time = dt.datetime.now()
        result = func(*args,**kwargs)
        end_time = dt.datetime.now()

        return result, end_time - start_time
    return func_with_runtime

def get_locals_dict(args, kwargs, func):
    """Gets dictionary of all variables in @func with specified @args and @kwargs"""

    iparams = inspect.signature(func).parameters

    locals_dict = dict(kwargs)
    for name, val in zip(iparams, args):
        if name not in locals_dict:
            locals_dict[name] = val
    for name, val in iparams.items():
        if name not in locals_dict and val.default != inspect._empty:
            locals_dict[name] = val.default

    return locals_dict

###########################
"""Main Decorator Function"""
###########################

def save_data(ignore=set()):
    """Decorator for saving and storing data using the @write_data and @retrieve_data functions

    To skip loading and overwrite existing data, set an 'overwrite' keyword argument in the
    function to True.

    To call off the whole process of reading and writing data, set a 'break_saving' keyword
    argument in the function to True.

    @ignore is a set of parameters to ignore in the data readout"""

    def save_data_wrapper(data_gen_func):
        @functools.wraps(data_gen_func)
        def func_with_data_saving(*args, overwrite = False, suppress = False, **kwargs):
            nonlocal ignore

            locals_dict = get_locals_dict(args, kwargs, data_gen_func)
            if 'break_saving' in kwargs and kwargs['break_saving']:
                if not suppress:
                    print(f'All data saving and retrieval turned off for {data_gen_func.__name__}')
                return data_gen_func(*args, **kwargs)
            ignore.add('break_saving')
            parent_dir = data_gen_func.__name__ + '_data'
            passed_vars = {'overwrite': overwrite,
                           'suppress': suppress}
            ignore = ignore.union(set(passed_vars.keys()))
            locals_dict = {key: val for key, val in locals_dict.items() if key not in ignore}
            data = retrieve_data(locals_dict, parent_dir=parent_dir, assume_nones=True, raise_errors = False,
                                 **passed_vars)
            if data is None or overwrite:
                data, runtime = runtime_decor(data_gen_func)(*args, **kwargs)
                write_data(data, locals_dict, parent_dir=parent_dir, runtime = runtime, **passed_vars)
                return data
            else:
                return data

        return func_with_data_saving

    return save_data_wrapper

###########################
"""Data Cleanup"""
###########################

def clear_data(parent_dir,date = dt.date.today(), suppress = False, deletion_dict = None):
    """Clears data from @parent_dir for specified @date

    If date == 'ALL', removes entire directory

    If @suppress, printouts not executed

    @deletion_dict is a dictionary describing all entries to be deleted on the specified dates.
    If None, then all data is deleted.

    To do: Rename data files to get rid of gaps
    """

    if not os.path.exists(parent_dir):
        raise CacheException(f'Directory {parent_dir} does not exist')

    reg_path = get_registry_path(parent_dir)

    if not os.path.exists(reg_path):
        raise CacheException(f'Directory {parent_dir} does not have a registry')

    registry = pd.read_csv(reg_path)

    if deletion_dict is not None:
        deletion_dict_subset = retrieve_info_row(deletion_dict,registry, assume_nones = False,
                                                 raise_errors = False, allow_duplicates= True,
                                                 suppress = True)
        in_deletion_dict = registry['Data Number'].isin(deletion_dict_subset['Data Number'])

    else:
        in_deletion_dict = registry.index == registry.index

    if date == 'ALL':
        has_date = registry.index == registry.index
    else:
        has_date = pd.to_datetime(registry['Date']).apply(lambda x: x.date()) == date

    to_delete_bools = has_date & in_deletion_dict

    if (~to_delete_bools).all():
        if not suppress:
            print('No data to delete')
        return

    to_delete = registry[to_delete_bools]
    new_registry = registry[~to_delete_bools].sort_values(by = 'Data Number').set_index(['Data Number','Date'])
    if new_registry.empty:
        shutil.rmtree(parent_dir)
        if not suppress:
            print('No data remaining\n')
            print(f'Removed directory {parent_dir}')
        return
    new_registry.to_csv(reg_path)
    for data_num in to_delete['Data Number']:
        data_path = os.path.join(parent_dir,f'{data_num}.csv')
        if not suppress:
            print(f'Deleting {data_path}')
        os.remove(data_path)

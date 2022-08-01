from __future__ import print_function
from datetime import datetime
#import yaml  #assume these are not available.
#import semantic_version #assume these are not available.
import re
import sys
import os

def eprint(*args, **kwargs):#Print to stderr
    print(*args, file=sys.stderr, **kwargs)

citations_dict = {}
checkpoint_log_filename = "CiteSoftwareCheckpointsLog.txt"
consolidated_log_filename = "CiteSoftwareConsolidatedLog.txt"
validate_on_fly = True#Flag.  If set to true, argument names will be checked in real time, and invalid argument names will result in a printed warning to the user
valid_optional_fields = ["version", "cite", "author", "doi", "url", "encoding", "misc"]
valid_required_fields = ['timestamp', 'unique_id', 'software_name']

#The module_call_cite function is intended to be used as a decorator.
#It is similar to the example "decorator_maker_with_arguments" at https://www.datacamp.com/community/tutorials/decorators-python
#To find the example, search for "decorator_maker_with_arguments" at the above link.
#function "inner" below is named 'decorator' in the above link and 'wrapper' below is named 'wrapper' in the above link.
def module_call_cite(unique_id, software_name, write_immediately=False, **add_args):
    #the unique_id and the software_name are the only truly required args.
    #Optional args are: ["version", "cite", "author", "doi", "url", "encoding", "misc"]
    #Every arg must be a string.
    def inner(func):
        def wrapper(*args, **kwargs):
            add_citation(unique_id, software_name, write_immediately, **add_args)
            result = func(*args, **kwargs)
            return result
        return wrapper
    return inner

#The after_call_compile_checkpoints_log function is intended to be used as a decorator.
#It is similar to the example "decorator_maker_with_arguments" at https://www.datacamp.com/community/tutorials/decorators-python
#To find the example, search for "decorator_maker_with_arguments" at the above link.
#function "inner" below is named 'decorator' in the above link and 'wrapper' below is named 'wrapper' in the above link.
def after_call_compile_checkpoints_log(file_path="", empty_checkpoints=True):
    def inner(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            compile_checkpoints_log(file_path=file_path, empty_checkpoints=empty_checkpoints)
            return result
        return wrapper
    return inner

#The after_call_compile_consolidated_log function is intended to be used as a decorator.
#It is similar to the example "decorator_maker_with_arguments" at https://www.datacamp.com/community/tutorials/decorators-python
#To find the example, search for "decorator_maker_with_arguments" at the above link.
#function "inner" below is named 'decorator' in the above link and 'wrapper' below is named 'wrapper' in the above link.
def after_call_compile_consolidated_log(file_path="", compile_checkpoints=True):
    def inner(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            compile_consolidated_log(file_path=file_path, compile_checkpoints=compile_checkpoints)
            return result
        return wrapper
    return inner

#The import_cite function is intended to be used at the top of a sofware module.
def import_cite(unique_id, software_name, write_immediately=False, **kwargs):
    add_citation(unique_id, software_name, write_immediately, **kwargs)

#The add_citation function is the method which actually adds a citation.
def add_citation(unique_id, software_name, write_immediately=False, **kwargs):
    new_entry = {'unique_id' : unique_id, 'software_name' : software_name, 'timestamp' : get_timestamp()}
    for key in kwargs:
        if validate_on_fly:
            if not key in valid_optional_fields:
                eprint("Warning, " + key + " is not an officially supported argument name.  Use of alternative argument names is strongly discouraged.")
        if type(kwargs[key]) is not list:#Make sure single optional args are wrapped in a list
            kwargs[key] = [kwargs[key]]
        new_entry[key] = kwargs[key]
    if unique_id in citations_dict:#Check for duplicate entries(e.g. from calling the same function twice)
        citations_dict[unique_id] = compare_same_id(citations_dict[unique_id], new_entry)
    else:
        citations_dict[unique_id] = new_entry
    if write_immediately == True:
        compile_checkpoints_log()
       
def compile_checkpoints_log(file_path="", empty_checkpoints=True):
    with open(file_path + checkpoint_log_filename, 'a') as file:
        write_dict_to_output(file, citations_dict)
    if empty_checkpoints==True:
        citations_dict.clear()

def compile_consolidated_log(file_path="", compile_checkpoints=True):
    if compile_checkpoints == True:
        compile_checkpoints_log()
    print("Warning: CiteSoftLocal cannot make a consolidated log. Citations have been exported to CiteSoftwareCheckpointsLog.txt")
    # consolidated_dict = {}
    # if consolidated_log_filename in os.listdir(): #check if the file exists already.
        # consolidated_log_exists = True
    # else:
        # consolidated_log_exists = False
    # if consolidated_log_exists == True: #can only read file if it exists.
        # with open(file_path + consolidated_log_filename, "r") as file:
            # yaml_file_contents = yaml.safe_load_all(file)
            # for yaml_document in yaml_file_contents:
                # if yaml_document != None: #This is for 'blank' documents of "---" with nothing after that symbol.
                    # for citation_entry in yaml_document:
                        # id = citation_entry["unique_id"]
                        # if id in consolidated_dict:
                            # consolidated_dict[id] = compare_same_id(consolidated_dict[id], citation_entry)
                        # else:
                            # consolidated_dict[id] = citation_entry
    # if checkpoint_log_filename in os.listdir(): #check if the file exists already.
        # checkpoint_log_exists = True
    # else:
        # checkpoint_log_exists = False
    # if checkpoint_log_exists == True: #can only read file if it exists.
        # with open(checkpoint_log_filename, 'r') as file:
            # yaml_file_contents = yaml.safe_load_all(file)
            # for yaml_document in yaml_file_contents:
                # if yaml_document != None: #This is for 'blank' documents of "---" with nothing after that symbol.
                    # for citation_entry in yaml_document:
                        # id = citation_entry["unique_id"]
                        # if id in consolidated_dict:
                            # consolidated_dict[id] = compare_same_id(consolidated_dict[id], citation_entry)
                        # else:
                            # consolidated_dict[id] = citation_entry
    # with open(consolidated_log_filename, 'w') as file:
        # write_dict_to_output(file, consolidated_dict)

#Takes a dictionary, converts it to CiteSoft-compatible YAML, and writes it to file
def write_dict_to_output(file, dictionary):
    file.write('---\n')
    for key,dict in dictionary.items():
        file.write('-\n')
        for s in valid_required_fields:
            file.write('    ' + s + ': >-\n')
            file.write('    '*2 + dict[s] + '\n')
        for subkey in dict:
            if subkey not in valid_required_fields:
                file.write('    ' + subkey + ':\n')
                if type(dict[subkey]) is list:
                    for i in dict[subkey]:
                        file.write('    '*2 + '- >-\n')
                        file.write('    '*3 + i + '\n')
                else:
                    file.write('    '*2 + '- >-\n')
                    file.write('    '*3 + dict[subkey] + '\n')

#Helper Functions

#Returns a string of the current time in the ISO 8601 format (YYYY-MM-DDThh:mm:ss).
def get_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    return timestamp

#Compares two entries
#Returns : The entry which should be kept
def compare_same_id(old_entry, new_entry):
    return new_entry #CiteSoftLocal will not do comparisons. It will just return the new_entry.
    # old_has_version = "version" in old_entry
    # new_has_version = "version" in new_entry
    # if old_has_version and new_has_version:#If both entries have a version, compare them return and the return the greater(newer) version
        # old_ver_str = str(old_entry["version"][0])
        # new_ver_str = str(new_entry["version"][0])
        # #Initialize variables, assume strings are valid unless parsing fails
        # old_ver_semver_valid = True
        # new_ver_semver_valid = True
        # decimal_regex_str = "^[0-9]+\.[0-9]+$"#Match string with decimal point enclosed by at least one number on either side
        # if re.match(decimal_regex_str, old_ver_str):
            # old_ver_str += '.0'#To ensure semantic version parser handles a decimal value correctly
        # if re.match(decimal_regex_str, new_ver_str):
            # new_ver_str += '.0'#To ensure semantic version parser handles a decimal value correctly
        # try:
            # old_sv = semantic_version.Version(old_ver_str)
        # except ValueError:
            # old_ver_semver_valid = False
        # try:
            # new_sv = semantic_version.Version(new_ver_str)
        # except:
            # new_ver_semver_valid = False
        # if old_ver_semver_valid and new_ver_semver_valid:#If both entries have a valid SemVer version, keep the older one only if it's greater. Else, keep the newer one.
            # if old_sv > new_sv:
                # return old_entry
            # else:
                # return new_entry
        # elif old_ver_semver_valid:#If only the old entry has a valid SemVer version, keep it
            # return old_entry
        # elif new_ver_semver_valid:#If only the new entry has a valid SemVer version, keep it
            # return new_entry
        # else:
            # #Version comparison failed, use alphanumeric comparison
            # if old_ver_str > new_ver_str:
                # return old_entry
            # else:
                # return new_entry
    # elif old_has_version and not new_has_version:#If old entry has a version and the new entry doesn't, the entry with a version takes precedence
        # return old_entry
    # elif not old_has_version and new_has_version:#Likewise, if new entry has a version and the old entry doesn't, the entry with a version takes precedence
        # return new_entry
    # else:#If neither entry has a version, save the new entry
        # return new_entry

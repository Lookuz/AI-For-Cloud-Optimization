import csv
import sys

def load_mapping(userdept_file):
    mega_dict = {}
    with open(userdept_file, "r") as csvf:
        csv_r = csv.reader(csvf)
        for row in csv_r:
            mega_dict.update({row[0]: row[1]})
    return mega_dict

def search(mega_dict, user_search):
    try:
        return mega_dict[user_search]
    except Exception as e:
        return None

def main(userdept_file, user_search):
    mega_dict = load_mapping(userdept_file)
    print(search(mega_dict, user_search)) 

if __name__ == '__main__':
    USER_DEPT_FILE = sys.argv[1].rstrip()
    USER_SEARCH = sys.argv[2].rstrip()
    main(USER_DEPT_FILE, USER_SEARCH)

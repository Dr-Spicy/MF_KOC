# MeetFresh_KOC
## clean.py: 
Merge multiple JSON files and only keep unique elements.
- How to use:
    ```shell
    python clean.py -i [input_folder] -o [output_folder] -e [is_extention]
    ```
- Parameter:
    - input_folder: the folder path where your original JSON files are.
    - output_folder: the folder path where your output JSON will be.
    - is_extention: enter "y" if your JSON files are collected by Google Chrome extension\
    otherwise, enter "n". If "y", you cannot have other JSON files under your input_folder.
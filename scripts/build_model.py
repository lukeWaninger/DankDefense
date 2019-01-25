"""
The "train -> validate -> predict -> submit" pipline script.
"""

import jsonschema
import argparse
import yaml

config_schema = """
type: object
properties:
    instance_type:
        type: string
    features:
        type: list
        items:
            type: string
    validation:
        type: string
    predict:
        type: object
        parameters:
            name:
                type: string
            submit:
                type: boolean
    parameter_tuning:
        type: object
        parameters:
            search_type:
                type: string
                enum:
                    - grid
                    - stagewise
            parameters:
                type: list
                items:
                    type: object
                    parameters:
                        name:
                            type: string
                        values:
                            type: array
                            uniqueItems: true
"""

def boot_instance():
    pass

def terminate_instance():
    pass

def main():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('config_file', type=str, help='relative path to thee config file')

    args = parser.parse_args()
    config_file_path = args.config_file



if __name__ == '__main__':
    main()

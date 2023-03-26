import os
import click

FILE_HEAD = """/*
    ********************************************************
    THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT
    ********************************************************
*/
#pragma once

"""

FILE_TAIL = """"""

@click.command()
@click.option('--source-path', '-s', required=True, help='Path to shader source file')
@click.option('--header-path', '-o', required=True, help='Path to header folder')
def to_header(source_path: str = None, header_path: str = None):
    file = open(source_path, 'r')
    lines = file.readlines()
    lines = ''.join(lines)
    file.close()

    header = f'static const char* {os.path.basename(source_path).replace(".", "_")} = R"('
    header_tail = ')";'
    for char in lines:
        header += char
    header += header_tail

    out_file_path = os.path.join(header_path, os.path.basename(source_path).split('.')[0])
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)
    
    out_file_name = os.path.basename(source_path) + '.h'
    with open(os.path.join(out_file_path, out_file_name), 'w') as out_file:
        out_file.writelines([FILE_HEAD, header, FILE_TAIL])

if __name__ == '__main__':
    to_header()
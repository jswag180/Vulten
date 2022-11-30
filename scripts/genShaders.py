import os
import subprocess
import click
import itertools
import struct

PREPRE_COMMAND = '//#/'

SHADER_HEADER_HEAD = """/*
    ********************************************************
    THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT
    ********************************************************
*/
#pragma once
#include <vector>
#include <stdint.h>

namespace shader{
"""

SHADER_HEADER_TAIL = """
}
"""

typeToIntDict = {
    'float' : 0,
    'int' : 1,
    'uint' : 2,
    'int64_t' : 3,
    'uint64_t' : 4,
    'int8_t' : 5,
    'uint8_t' : 6,
    'double' : 7,
}

intToTypeDic = inv_map = {v: k for k, v in typeToIntDict.items()}

@click.command()
@click.option('--source-path', '-s', required=True, help='Path to shader source file')
@click.option('--header-path', '-o', required=True, help='Path to header folder')
@click.option('--build-types', '-t', required=False, default='float', help='Types to build shader for')
@click.option('--build-args', '-a', required=False, default=None, help='Args to pass to compiler')
def genShader(source_path: str = None, header_path: str = None, build_types: str = None, build_args: str = None):
    file = open(source_path, 'r')
    lines = file.readlines()

    build_types_list = build_types.split(',')

    if lines[0].startswith(PREPRE_COMMAND):
        command = lines[0].removeprefix(PREPRE_COMMAND).replace('\n','').replace(' ', '').split(',')
        args = command[1:]
        command = command[0]



        if command == 'types':
            wheels = [build_types_list] * int(args[0])
            combos = itertools.product(*wheels)

            out_file_path = os.path.join(header_path, os.path.basename(source_path).removesuffix('.comp'))
            if not os.path.exists(out_file_path):
                os.makedirs(out_file_path)

            shader_vecs = []

            for combo in combos:
                if not build_args == None:
                    run = build_args.split(' ')
                else:
                    run = []
                out_file_name = os.path.basename(source_path).removesuffix('.comp')
                for resWheel in range(len(combo)):
                    run.append(f'-DTYPE_{resWheel}=' + combo[resWheel])
                    run.append(f'-DTYPE_NUM_{resWheel}=' + str(typeToIntDict[combo[resWheel]]))
                    out_file_name = out_file_name + '_' + combo[resWheel]
                
                out_file_name = out_file_name + ".spv"
                run = subprocess.run(['glslangValidator', '-V', source_path, '-o', os.path.join(out_file_path, out_file_name), *run], capture_output=True, check=False)
                if(run.returncode != 0):
                    print(f'Shader: {out_file_name} did not compile for type(s) {combo}')
                    print('stderr:')
                    print(run.stderr.decode("utf-8") )
                    print('stdout:')
                    print(run.stdout.decode("utf-8") )
                    exit(-1)

                shader_define = '   #define ' + out_file_name.removesuffix('.spv').upper() + '\n'
                vector_head = '   static const std::vector<uint32_t> ' + out_file_name.removesuffix('.spv') + ' = {'
                vector_tail = '};'

                with open(os.path.join(out_file_path, out_file_name), 'rb') as spv:
                    spv_bytes = spv.read()
                    hex_int_string = ''
                    for i in range(0, len(spv_bytes), 4):
                        if i == 0:
                            hex_int_string = str(struct.unpack('@I', spv_bytes[i:i+4])[0])
                        else:
                            hex_int_string = hex_int_string + ',' + str(struct.unpack('@I', spv_bytes[i:i+4])[0])
                
                shader_vecs.append(shader_define + vector_head + hex_int_string + vector_tail + '\n')
                
            out_header_lines = [SHADER_HEADER_HEAD, *shader_vecs, SHADER_HEADER_TAIL]

            with open(os.path.join(out_file_path, os.path.basename(source_path).removesuffix('.comp') + '.h'), 'w') as out_header_file:
                out_header_file.writelines(out_header_lines)
                
        else:
            print(f'Commands: {command} is not found')

if __name__ == '__main__':
    genShader()
    
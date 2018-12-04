#!/usr/bin/env python
"""
Convert jupyter notebook to sphinx gallery notebook styled examples.
Needs pypandoc (`pip install pypandoc`).

Usage:
    nb2sphx NOTEBOOK

"""

import json
import pypandoc as pdoc

from km3pipe import version


def convert_ipynb_to_gallery(file_name):
    """
    Blatantly stolen + adapted from
    https://gist.github.com/wuhuikai/4a7ceb8bc52454e17a4eb8327d538d85

    """
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md')
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                commented_source = '\n'.join([
                    '# ' + x for x in rst_source.split('\n')
                ])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

    open(file_name.replace('.ipynb', '.py'), 'w').write(python_file)


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)
    fname = args['NOTEBOOK']
    convert_ipynb_to_gallery(fname)


if __name__ == '__main__':
    main()

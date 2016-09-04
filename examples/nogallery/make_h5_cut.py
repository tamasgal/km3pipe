#!/usr/bin/env python
"""Apply a cut on a table & copy to a new file.

You can use pytable's expression syntax to apply logical conditions
like

    sqrt(pos_x**2 + pos_y**2) <= 150.0

using numexpr-like syntax to filter events from a table.
To only grab the events which fullfill that condition, use `my_table.where()`,
or the related functions `read_where` and `append_where`.
"""

import sys
import tables as tb

infile = sys.argv[-2]
outfile = sys.argv[-1]

h5in = tb.open_file(infile, 'r')
h5out = tb.open_file(outfile, 'a')

tab_in = h5in.root.mva
tab_desc = tab_in.description
filt = tb.Filters(complevel=5)
tab_out = h5out.create_table('/', 'mva', description=tab_desc, filters=filt)

expr = 'sqrt(tf_r_x**2 + tf_r_y**2) <= 150.0'

tab_in.append_where(tab_out, expr)
tab_in.close()
tab_out.close()

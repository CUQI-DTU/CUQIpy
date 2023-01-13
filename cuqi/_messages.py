""" This module defines messages (errors, warnings, etc) that are used throughout CUQIpy. The values of the following variables are not meant to be changed by the user."""

_disable_warning_msg = lambda module_name: "To disable "+\
"warnings for a given module or library, "+\
"you can use the method `warnings.filterwarnings`,"+\
" e.g.: warnings.filterwarnings(action='ignore', "+\
f"module=r'"+module_name+"')."

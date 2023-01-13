""" This module defines variables that are used throughout CUQIpy. The values of these variables are not meant to be changed by the user."""

_disable_warning_msg = lambda module_name: "To disable "+\
"warnings for a given module or library, "+\
"you can use the method `warnings.filterwarnings`,"+\
" e.g.: warnings.filterwarnings(action='ignore', "+\
f"module=r'"+module_name+"')."

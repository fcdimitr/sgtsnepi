project('sgtsnepi', 'cpp',
        version : '1.0.2',
        license : 'GPL-3.0-or-later',
        meson_version : '>=0.53.0',
        default_options : ['optimization=2',
                           'debug=false',
                           'cpp_std=c++11',
                           'default_library=both'])
                           # 'prefix=' + meson.current_source_dir()])

fs  = import('fs')              # filesystem module
vso = '0'                       # shared library version ID

# ========== compiler options

cc = meson.get_compiler('cpp')

# compiler properties and Cilk support
is_icpc      = cc.get_id().contains('intel')
use_opencilk = not is_icpc \
               and cc.has_argument('-fopencilk')
use_cilkplus = not use_opencilk \
               and (is_icpc or cc.has_argument('-fcilkplus'))

if not (use_opencilk or use_cilkplus)
  cc_str = cc.get_id() + ' ' + cc.version()
  error('The specified compiler (' + cc_str + ') does not support Cilk.')
endif

# relevant Cilk headers
cc.has_header('cilk/cilk.h'         , required : true)
cc.has_header('cilk/cilk_api.h'     , required : true)




# icpc-specific flags
if is_icpc
  add_project_arguments(['-wd3947,3946,10006,3950'], language : 'cpp')
  add_project_link_arguments(['-lirc','-limf','-lsvml'], language : 'cpp')
endif

# Cilk flags
if use_opencilk
  add_project_arguments(['-fopencilk','-DOPENCILK'], language : 'cpp')
  add_project_link_arguments(['-fopencilk'], language : 'cpp')
elif use_cilkplus
  add_project_arguments(['-fcilkplus'], language : 'cpp')
  add_project_link_arguments(['-lcilkrts'], language : 'cpp')
endif

# add flag for showing times
if get_option('print_times')
  add_project_arguments(['-DPRINT_DEBUG_TIME'], language : 'cpp')
endif

# compiler optimizations for host computer
if get_option('tune_native')
  if is_icpc
    add_project_arguments(['-xHost'], language : 'cpp')
  else # clang/gcc
    add_project_arguments(['-march=native','-mtune=native'], language : 'cpp')
  endif
endif

# ========== dependencies

# TODO When looking for library dependencies, allow the user to specify a
# static/shared library object to use instead of whatever system package is
# found by `dependency()`.  See https://mesonbuild.com/Dependencies.html

libs_dep          = []
libs_dep_dict     = {}
str_lib_not_found = 'The @0@ library is required.'

# Search for all required dependencies, first using `dependency()` and then
# using `cc.find_library()` (if `dependency()` pkg-config or CMake search
# fails).  Format per dependency: [<lib pkg name>, <lib name>, [<lib headers>]].
required_lib_deps = [['fftw3'     , 'FFTW3'                  , ['fftw3.h']],
                     ['fftw3f'    , 'FFTW3f'                 , ['fftw3.h']]]
foreach l : required_lib_deps
  buf_dep = dependency(l[0], required : false)
  if not buf_dep.found()
    buf_dep = cc.find_library(l[0], required : true, has_headers : l[2])
  endif
  if not buf_dep.found()
    error(str_lib_not_found.format(l[1]))
  endif
  libs_dep      += buf_dep
  libs_dep_dict += {l[0] : buf_dep}
endforeach

# ========== build targets

# source files
subdir('src')
subdir('csb')

# SG-t-SNE-Pi library
sgtsnepi_lib = library('sgtsnepi', [sgtsnepi_src],
                       dependencies : [libs_dep, csb_dep],
                       include_directories : csb_inc_dir,
                       install : true, install_dir : 'lib',
                       soversion : vso)


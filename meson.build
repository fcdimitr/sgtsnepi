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

# relevant Cilk headers
if use_opencilk
  cc.has_header('cilk/cilk.h'         , required : true)
  cc.has_header('cilk/cilk_api.h'     , required : true)
endif




# icpc-specific flags
if is_icpc
  add_project_arguments(['-wd3947,3946,10006,3950'], language : 'cpp')
  add_project_link_arguments(['-lirc','-limf','-lsvml'], language : 'cpp')
endif

# Cilk flags
if use_opencilk
  add_project_arguments(['-fopencilk','-DOPENCILK'], language : 'cpp')
  add_project_link_arguments(['-fopencilk'], language : 'cpp')
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

# parallel FFTW library
fftw_par_dep = dependency('fftw3_' + get_option('fftw_parallel_lib'),
                          required : true)
if not fftw_par_dep.found()
  fftw_par_dep = cc.find_library('fftw3_' + get_option('fftw_parallel_lib'),
                                 required : true)
endif
if not fftw_par_dep.found()
  error('Could not find fftw3_' + get_option('fftw_parallel_lib') + ' library.')
endif

fftwf_par_dep = dependency('fftw3f_' + get_option('fftw_parallel_lib'),
                          required : true)
if not fftwf_par_dep.found()
  fftwf_par_dep = cc.find_library('fftw3f_' + get_option('fftw_parallel_lib'),
                                 required : true)
endif
if not fftwf_par_dep.found()
  error('Could not find fftw3f_' + get_option('fftw_parallel_lib') + ' library.')
endif

libs_dep = [fftw_par_dep] + [fftwf_par_dep] + libs_dep 
# NOTE -lfftw3_<parallel> should precede -lfftw3
libs_dep_dict += {'fftw3_parallel' : fftw_par_dep,
                  'fftw3f_parallel' : fftwf_par_dep}


# ========== build targets

# source files
subdir('src')
#subdir('csb')

# SG-t-SNE-Pi library
sgtsnepi_lib = library('sgtsnepi', [sgtsnepi_src],
                       dependencies : [libs_dep],
                       install : true, install_dir : 'lib',
                       soversion : vso)


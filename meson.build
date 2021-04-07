project('sgtsnepi', 'cpp',
        version : '1.0.2',
        license : 'GPL-3.0-or-later',
        meson_version : '>=0.53.0',
        default_options : ['optimization=2',
                           'debug=false',
                           'cpp_std=c++11',
                           'default_library=both',
                           'prefix=' + meson.current_source_dir()])

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
cc.has_header('cilk/reducer_opadd.h', required : true)
cc.has_header('cilk/reducer_max.h'  , required : true)

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
required_lib_deps = [['fftw3'    , 'FFTW3'                 , ['fftw3.h']],
                     ['metis'    , 'METIS'                 , ['metis.h']],
                     ['tbbmalloc', 'TBB scalable allocator', ['tbb/scalable_allocator.h']],
                     ['flann'    , 'FLANN'                 , ['flann/flann.h']],
                     ['lz4'      , 'LZ4'                   , ['lz4.h']]]
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
if not ['', 'none'].contains(get_option('fftw_parallel_lib'))
  fftw_par_dep = dependency('fftw3_' + get_option('fftw_parallel_lib'),
                            required : false)
  if not fftw_par_dep.found()
    fftw_par_dep = cc.find_library('fftw3_' + get_option('fftw_parallel_lib'),
                                   required : false)
  endif
  if not fftw_par_dep.found()
    warning('Could not find fftw3_' + get_option('fftw_parallel_lib') + 'library.' +
            ' Using single-threaded FFTW3 execution.')
  endif
else
  fftw_par_dep = dependency('', required : false)
endif
if not fftw_par_dep.found()
  add_project_arguments(['-DFFTW_NO_PARALLEL'], language : 'cpp')
endif
libs_dep = [fftw_par_dep] + libs_dep # NOTE -lfftw3_<parallel> should precede -lfftw3
libs_dep_dict += {'fftw3_parallel' : fftw_par_dep}

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

# conventional t-SNE implementation
tsnepi_exe = executable('tsnepi', tsnepi_src,
                        link_with : sgtsnepi_lib,
                        dependencies : [libs_dep_dict['flann'],
                                        libs_dep_dict['lz4']],
                        install : true, install_dir : 'bin',
                        install_rpath : get_option('prefix') / 'lib')

# demos
if get_option('enable_demos')
  demo_stochastic_matrix = executable('demo_stochastic_matrix',
                                      demo_stochastic_matrix_src,
                                      link_with : sgtsnepi_lib,
                                      install : true, install_dir : 'bin',
                                      install_rpath : get_option('prefix') / 'lib')
  demo_perplexity_equalization = executable('demo_perplexity_equalization',
                                            demo_perplexity_equalization_src,
                                            link_with : sgtsnepi_lib,
                                            dependencies : [libs_dep_dict['flann'],
                                                            libs_dep_dict['lz4']],
                                            install : true, install_dir : 'bin',
                                            install_rpath : get_option('prefix') / 'lib')
endif

# test
test_modules = executable('test_modules', test_modules_src,
                          link_with : sgtsnepi_lib,
                          install : true, install_dir : 'bin',
                          install_rpath : get_option('prefix') / 'lib')
test('SG-t-SNE-Pi test modules', test_modules, is_parallel : false)


# ========== Matlab interface

if get_option('enable_matlab')

  mroot = get_option('matlabroot')

  # MEX compilation flags
  if host_machine.system() == 'darwin' # macOS
    mexext   = 'mexmaci64'
    mexarch  = 'maci64'
    mexrpath = []
    mexsym   = ['-bundle',
                '-Wl,-exported_symbols_list,' + mroot / 'extern' / 'lib' / mexarch / 'mexFunction.map']
  else                                 # Linux
    if host_machine.system() != 'linux'
      warning('Not sure how to configure MEX compilation for detected system' \
              + ' (' + host_machine.system() + ').  Using Linux configuration.')
    endif
    mexext   = 'mexa64'
    mexarch  = 'glnxa64'
    mexrpath = ['-Wl,-rpath=' + mroot / 'bin' / mexarch]
    if not is_icpc and use_cilkplus
      mexrpath += ['-Wl,-rpath=' + get_option('dir_libcilkrts')]
    endif
    mexsym   = ['-shared',
                '-Wl,--version-script,' + mroot / 'extern' / 'lib' / mexarch / 'mexFunction.map']
  endif
  mex_cpp_args  = ['-I' + mroot / 'extern' / 'include'] + mexrpath + ['-fPIC']
  mex_link_args = ['-L' + mroot / 'bin' / mexarch, '-fno-common'] \
                  + mexsym + ['-lmx', '-lmex', '-lmat']

  # build MEX executables
  mex_bin_all = []
  foreach mex_src : mex_src_all
    # <path/to/filename>_mex.cpp --> <filename>.<mexext>
    mex_bin = fs.name('@0@'.format(mex_src[0]).split('_mex')[0])
    mex_bin = fs.replace_suffix(mex_bin, '.'+mexext)
    mex_bin_all += mex_bin
    executable(mex_bin, mex_src[0],
               cpp_args : mex_cpp_args,
               link_args : mex_link_args,
               link_with : sgtsnepi_lib,
               dependencies : mex_src[1],
               build_rpath : meson.current_build_dir(),
               install : true, install_dir : 'matlab',
               install_rpath : get_option('prefix') / 'lib')
  endforeach

  # install M-files
  install_subdir('matlab', install_dir : '', exclude_files : mex_bin_all)

endif

# ========== documentation

subdir('docs')

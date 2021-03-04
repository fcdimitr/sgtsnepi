project('sgtsnepi', 'cpp',
        version : '1.0.2',
        license : 'GPL-3.0-or-later',
        meson_version : '>=0.50.0',
        default_options : ['optimization=2',
                           'debug=false',
                           'cpp_std=c++11',
                           'default_library=both',
                           'prefix=' + meson.current_source_dir()])

fs  = import('fs')              # filesystem module
vso = '0'                       # shared library version ID

# ========== compiler options

cc = meson.get_compiler('cpp')

is_icpc      = cc.get_id().contains('intel')
use_opencilk = not is_icpc \
               and cc.has_argument('-fopencilk') and cc.has_header('cilk/cilk.h')
use_cilkplus = not use_opencilk \
               and (is_icpc or cc.has_argument('-fcilkplus')) \
               and cc.has_header('cilk/cilk.h')

if not (use_opencilk or use_cilkplus)
  cc_str = cc.get_id() + ' ' + cc.version()
  error('The specified compiler (' + cc_str + ') does not support Cilk')
endif

if is_icpc                      # icpc-specific flags
  add_project_arguments(['-wd3947,3946,10006,3950'], language : 'cpp')
  add_project_link_arguments(['-lirc','-limf','-lsvml'], language : 'cpp')
endif

if use_opencilk                 # OpenCilk flags
  add_project_arguments(['-fopencilk','-DOPENCILK'], language : 'cpp')
  add_project_link_arguments(['-fopencilk'], language : 'cpp')
elif use_cilkplus               # Cilk Plus flags
  add_project_arguments(['-fcilkplus'], language : 'cpp')
  add_project_link_arguments(['-lcilkrts'], language : 'cpp')
endif

# ========== dependencies

fftw_dep = cc.find_library('fftw3', required : true)
if not fftw_dep.found()
  error('The FFTW3 library is required. ' +
        'Please issue `make dependencies` and retry.')
endif

fftw_par_lib_dir = get_option('fftw_parallel_lib_dir')
if not (fftw_par_lib_dir == '' or fs.is_absolute(fftw_par_lib_dir))
  error('The \'fftw_parallel_lib_dir\' option must be empty or an *absolute* path' +
        ' (input path: ' + fftw_par_libdir + ')')
endif
if fftw_par_lib_dir == ''
  fftw_par_lib_dir = []
endif
if get_option('fftw_parallel_lib') != ''
  fftw_par_dep = cc.find_library('fftw3_' + get_option('fftw_parallel_lib'),
                                 dirs : fftw_par_lib_dir, required : false)
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

metis_dep = cc.find_library('metis', required : true)
if not metis_dep.found()
  error('The METIS library is required. ' +
        'Please issue `make dependencies` and retry.')
endif

tbbmalloc_dep = cc.find_library('tbbmalloc', required : true)
if not tbbmalloc_dep.found()
  error('The TBB scalable malloc library is required. ' +
        'Please issue `make dependencies` and retry.')
endif

flann_dep = cc.find_library('flann', required : true,
                            has_headers : ['flann/flann.h'])
if not flann_dep.found()
  error('The FLANN library is required. ' +
        'Please issue `make dependencies` and retry.')
endif

lz4_dep = cc.find_library('lz4', required : true)

libs_dep = [fftw_par_dep, fftw_dep, metis_dep, tbbmalloc_dep, flann_dep, lz4_dep]


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
                        dependencies : [flann_dep, lz4_dep],
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
                                            dependencies : [flann_dep, lz4_dep],
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
    mexarch  = 'maxi64'
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


# ========== OpenCilk tools

if get_option('cilktool') == 'cilkscale' # ----- Cilkscale
  if not use_opencilk
    warning('Cilkscale instrumentation specified but not using OpenCilk compiler; skipping...')
  else
    arg_cs = '-fcilktool=cilkscale'
    def_cs = '-DCILKSCALE'
    sgtsnepi_lib_cs = library('sgtsnepi-cilkscale', [sgtsnepi_src],
                              dependencies : [libs_dep, csb_dep],
                              include_directories : csb_inc_dir,
                              cpp_args : [arg_cs, def_cs],
                              link_args : [arg_cs],
                              soversion : vso)
    sgtsnepi_lib_cs_bench = library('sgtsnepi-cilkscale-bench', [sgtsnepi_src],
                                    dependencies : [libs_dep, csb_dep],
                                    include_directories : csb_inc_dir,
                                    cpp_args : [arg_cs+'-benchmark', def_cs],
                                    link_args : [arg_cs],
                                    soversion : vso)
    if get_option('enable_demos')
      demo_stochastic_cs = executable('demo_stochastic_matrix_cs',
                                      demo_stochastic_matrix_src,
                                      link_with : sgtsnepi_lib_cs)
      demo_stochastic_cs_bench = executable('demo_stochastic_matrix_cs_bench',
                                            demo_stochastic_matrix_src,
                                            link_with : sgtsnepi_lib_cs_bench)
      demo_perplexity_cs = executable('demo_perplexity_equalization_cs',
                                      demo_perplexity_equalization_src,
                                      link_with : sgtsnepi_lib_cs,
                                      dependencies : [flann_dep, lz4_dep])
      demo_perplexity_cs_bench = executable('demo_perplexity_equalization_cs_bench',
                                            demo_perplexity_equalization_src,
                                            link_with : sgtsnepi_lib_cs_bench,
                                            dependencies : [flann_dep, lz4_dep])
    endif
  endif
endif

if get_option('cilktool') == 'cilksan' # ----- Cilksan
  if not use_opencilk
    warning('Cilksan instrumentation was specified but not using OpenCilk compiler; skipping...')
  else
    if not get_option('debug')
      warning('Cilksan instrumentation was specified but not building in debug mode!')
    endif
    args_san = ['-fsanitize=cilk']
    sgtsnepi_lib_san = library('sgtsnepi-cilksan', [sgtsnepi_src],
                               dependencies : [libs_dep, csb_dep],
                               include_directories : csb_inc_dir,
                               cpp_args : args_san,
                               link_args : args_san + ['-shared-libasan'],
                               soversion : vso)
    test_modules_san = executable('test_modules_cilksan', test_modules_src,
                                  link_with : sgtsnepi_lib_san,
                                  cpp_args : args_san,
                                  link_args : args_san)
    if get_option('enable_demos')
      demo_stochastic_san = executable('demo_stochastic_matrix_cilksan',
                                       demo_stochastic_matrix_src,
                                       link_with : sgtsnepi_lib_san)
      demo_perplexity_san = executable('demo_perplexity_equalization_cilksan',
                                       demo_perplexity_equalization_src,
                                       link_with : sgtsnepi_lib_san,
                                       link_args : args_san,
                                       dependencies : [flann_dep, lz4_dep])
    endif
  endif
endif


# ========== Documentation

subdir('docs')

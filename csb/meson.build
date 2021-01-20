csb_wrapper_src = files(['csb_wrapper.cpp'])
csb_inc_dir     = include_directories('.')

csb_dep = declare_dependency(sources             : csb_wrapper_src,
                             include_directories : csb_inc_dir,
                             compile_args        : ['-DALIGN=64'])

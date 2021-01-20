if get_option('enable_documentation')

  doxygen = find_program('doxygen', required : false)

  if doxygen.found()
    doc_src = configure_file(input : 'doxygen.config', output : 'doxygen.config',
                             copy : true, install : false)
    doc_tgt = custom_target('documentation', command : [doxygen, doc_src],
                            input : doc_src, output : 'html',
                            install : true, install_dir : 'doc')
  else
    warning('Could not find doxygen; `documentation` target disabled.')
  endif

endif

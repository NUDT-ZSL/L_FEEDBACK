# CMake generated Testfile for 
# Source directory: /home/liang/feedback/uid40
# Build directory: /home/liang/feedback/uid40/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(canyon_acceptance "/usr/bin/python3" "/home/liang/feedback/uid40/tests/acceptance.py" "/home/liang/feedback/uid40/build/canyon_renderer" "/home/liang/feedback/uid40/scenes" "/home/liang/feedback/uid40/build/acceptance_out")
set_tests_properties(canyon_acceptance PROPERTIES  _BACKTRACE_TRIPLES "/home/liang/feedback/uid40/CMakeLists.txt;28;add_test;/home/liang/feedback/uid40/CMakeLists.txt;0;")

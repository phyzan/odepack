# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/senpy/Documents/repos/odepack/python

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/senpy/Documents/repos/odepack/python/build

# Include any dependencies generated for this target.
include CMakeFiles/odepack.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/odepack.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/odepack.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/odepack.dir/flags.make

CMakeFiles/odepack.dir/codegen:
.PHONY : CMakeFiles/odepack.dir/codegen

CMakeFiles/odepack.dir/pyode.cpp.o: CMakeFiles/odepack.dir/flags.make
CMakeFiles/odepack.dir/pyode.cpp.o: /home/senpy/Documents/repos/odepack/python/pyode.cpp
CMakeFiles/odepack.dir/pyode.cpp.o: CMakeFiles/odepack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/senpy/Documents/repos/odepack/python/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/odepack.dir/pyode.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/odepack.dir/pyode.cpp.o -MF CMakeFiles/odepack.dir/pyode.cpp.o.d -o CMakeFiles/odepack.dir/pyode.cpp.o -c /home/senpy/Documents/repos/odepack/python/pyode.cpp

CMakeFiles/odepack.dir/pyode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/odepack.dir/pyode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/senpy/Documents/repos/odepack/python/pyode.cpp > CMakeFiles/odepack.dir/pyode.cpp.i

CMakeFiles/odepack.dir/pyode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/odepack.dir/pyode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/senpy/Documents/repos/odepack/python/pyode.cpp -o CMakeFiles/odepack.dir/pyode.cpp.s

# Object files for target odepack
odepack_OBJECTS = \
"CMakeFiles/odepack.dir/pyode.cpp.o"

# External object files for target odepack
odepack_EXTERNAL_OBJECTS =

odepack.cpython-312-x86_64-linux-gnu.so: CMakeFiles/odepack.dir/pyode.cpp.o
odepack.cpython-312-x86_64-linux-gnu.so: CMakeFiles/odepack.dir/build.make
odepack.cpython-312-x86_64-linux-gnu.so: CMakeFiles/odepack.dir/compiler_depend.ts
odepack.cpython-312-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
odepack.cpython-312-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.a
odepack.cpython-312-x86_64-linux-gnu.so: CMakeFiles/odepack.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/senpy/Documents/repos/odepack/python/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module odepack.cpython-312-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/odepack.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/senpy/Documents/repos/odepack/python/build/odepack.cpython-312-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/odepack.dir/build: odepack.cpython-312-x86_64-linux-gnu.so
.PHONY : CMakeFiles/odepack.dir/build

CMakeFiles/odepack.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/odepack.dir/cmake_clean.cmake
.PHONY : CMakeFiles/odepack.dir/clean

CMakeFiles/odepack.dir/depend:
	cd /home/senpy/Documents/repos/odepack/python/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/senpy/Documents/repos/odepack/python /home/senpy/Documents/repos/odepack/python /home/senpy/Documents/repos/odepack/python/build /home/senpy/Documents/repos/odepack/python/build /home/senpy/Documents/repos/odepack/python/build/CMakeFiles/odepack.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/odepack.dir/depend


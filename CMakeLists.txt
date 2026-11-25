cmake_minimum_required(VERSION 3.20)
project(keel_control VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_CONTROL_NO_EXCEPTIONS_RTTI "Compile keel_control without C++ exceptions and RTTI" OFF)
option(KEEL_CONTROL_BUILD_TESTS "Build keel_control unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_CONTROL_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        869a5c07a5d77341e5f6acbac5f8e024b274ea08
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::poly)
    find_package(keel_poly QUIET CONFIG)
    if(TARGET keel::keel_poly AND NOT TARGET keel::poly)
        add_library(keel::poly ALIAS keel::keel_poly)
    endif()
endif()
if(NOT TARGET keel::poly)
    include(FetchContent)
    set(KEEL_POLY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../poly/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_POLY)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_POLY "${CMAKE_CURRENT_SOURCE_DIR}/../poly" CACHE PATH "Local keel_poly checkout")
    endif()
    if(KEEL_CONTROL_NO_EXCEPTIONS_RTTI)
        set(KEEL_POLY_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_poly
        GIT_REPOSITORY https://github.com/keelcpp/poly.git
        GIT_TAG        87429cfe8d8a3353b24ffdfa7385baba42c049ee
    )
    FetchContent_MakeAvailable(keel_poly)
endif()

if(NOT TARGET keel::linalg)
    find_package(keel_linalg QUIET CONFIG)
    if(TARGET keel::keel_linalg AND NOT TARGET keel::linalg)
        add_library(keel::linalg ALIAS keel::keel_linalg)
    endif()
endif()
if(NOT TARGET keel::linalg)
    include(FetchContent)
    set(KEEL_LINALG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../linalg/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_LINALG)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_LINALG "${CMAKE_CURRENT_SOURCE_DIR}/../linalg" CACHE PATH "Local keel_linalg checkout")
    endif()
    if(KEEL_CONTROL_NO_EXCEPTIONS_RTTI)
        set(KEEL_LINALG_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_linalg
        GIT_REPOSITORY https://github.com/keelcpp/linalg.git
        GIT_TAG        bfbb5449d06dae36a3bafe8f80700082d8201b01
    )
    FetchContent_MakeAvailable(keel_linalg)
endif()

add_library(keel_control STATIC
    src/numerics.cpp
    src/lyapunov.cpp
    src/riccati.cpp
    src/place.cpp
    src/state.cpp
    src/conversion.cpp
    src/discretize.cpp
    src/response.cpp
    src/freq.cpp
    src/hinfnorm.cpp
)
add_library(keel::control ALIAS keel_control)

target_link_libraries(keel_control
    PUBLIC  keel::keel
    PRIVATE keel::poly keel::linalg
)

target_compile_features(keel_control PUBLIC cxx_std_20)
set_target_properties(keel_control PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_control PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_control PRIVATE /W4 /utf-8)
    if(KEEL_CONTROL_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_control PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_control PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_control PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_CONTROL_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_control PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_control
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_CONTROL_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_control EXPORT keel_controlTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_controlTargets
    FILE keel_controlTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_control
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_controlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_controlConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_control
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_controlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_controlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_controlConfigVersion.cmake"
    DESTINATION lib/cmake/keel_control
)

if(KEEL_CONTROL_BUILD_TESTS)
    enable_testing()

    add_executable(test_control tests/test_control.cpp)
    target_link_libraries(test_control PRIVATE keel::control)
    if(MSVC)
        target_compile_options(test_control PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_control PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(COMMAND keel_enforce_guidelines)
        keel_enforce_guidelines(test_control
            SCOPE ALL
            BAN_RAW_INTS ALL
            BAN_RAW_FLOATS ALL
            WARNINGS OFF
        )
    endif()
    add_test(NAME test_control COMMAND test_control)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
